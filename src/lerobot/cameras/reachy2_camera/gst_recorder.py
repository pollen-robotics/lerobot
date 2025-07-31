import argparse
from threading import Lock, Thread
from typing import Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst, GstApp  # type: ignore [attr-defined]
from gst_signalling.utils import find_producer_peer_id_by_name


class GstRecorder:
    def __init__(
        self, signalling_host: str, signalling_port: int, peer_id: Optional[str] = None, peer_name: Optional[str] = None
    ) -> None:
        print("init GstRecorder")
        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls: Optional[Thread] = None
        self._left_image_lock = Lock()
        self._right_image_lock = Lock()

        self.pipeline = Gst.Pipeline.new("webRTC-recorder")
        self.source = Gst.ElementFactory.make("webrtcsrc")
        self._left_image: Optional[bytes] = None
        self._right_image: Optional[bytes] = None

        self.got_first_images = False

        if not self.pipeline:
            print("Pipeline could be created.")
            exit(-1)

        if not self.source:
            print(
                "webrtcsrc component could not be created. Please make sure that the plugin is installed \
                (see https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs/-/tree/main/net/webrtc)"
            )
            exit(-1)

        self.pipeline.add(self.source)

        if peer_id is None:
            peer_id = find_producer_peer_id_by_name(signalling_host, signalling_port, peer_name)
            print(f"found peer id: {peer_id}")

        self.source.connect("pad-added", self.webrtcsrc_pad_added_cb)
        signaller = self.source.get_property("signaller")
        signaller.set_property("producer-peer-id", peer_id)
        signaller.set_property("uri", f"ws://{signalling_host}:{signalling_port}")

    def new_sample(self, sink: GstApp.AppSink, udata: bool) -> Gst.FlowReturn:
        """Callback on 'new-sample' signal"""
        sample = sink.pull_sample()
        if isinstance(sample, Gst.Sample):
            buf = sample.get_buffer()
            if buf is None:
                print("Buffer is None")
                return Gst.FlowReturn.ERROR
            img = buf.extract_dup(0, buf.get_size())
            if udata:
                with self._left_image_lock:
                    self._left_image = img
            else:
                with self._right_image_lock:
                    self._right_image = img
        self.got_first_images = True
        return Gst.FlowReturn.OK

    def get_image(self, left: bool = False) -> Optional[bytes]:
        """Get the latest image from the appsrc (thread-safe, blocking lock)"""
        if left:
            with self._left_image_lock:
                img = self._left_image
                self._left_image = None
        else:
            with self._right_image_lock:
                img = self._right_image
                self._right_image = None
        return img

    def webrtcsrc_pad_added_cb(self, webrtcsrc: Gst.Element, pad: Gst.Pad) -> None:
        pad_name = pad.get_name()
        assert pad_name is not None, "Pad name should not be None"
        if pad_name.startswith("video"):
            print("video channel received: ", pad.get_name())
            videodepay = Gst.ElementFactory.make("rtph264depay")
            assert videodepay is not None
            queue = Gst.ElementFactory.make("queue")
            assert queue is not None
            h264parse = Gst.ElementFactory.make("h264parse")
            assert h264parse is not None
            decoder = Gst.ElementFactory.make("openh264dec")
            assert decoder is not None
            videoconvert = Gst.ElementFactory.make("videoconvert")
            assert videoconvert is not None
            jpegenc = Gst.ElementFactory.make("jpegenc")
            assert jpegenc is not None

            sink = Gst.ElementFactory.make("appsink")
            assert sink is not None
            caps = Gst.Caps.from_string("image/jpeg")
            sink.set_property("caps", caps)
            # we keep only the last frame
            sink.set_property("max-buffers", 1)
            sink.set_property("drop", True)
            sink.set_property("emit-signals", True)

            if pad_name.endswith("_0"):
                sink.connect("new-sample", self.new_sample, True)
            else:
                sink.connect("new-sample", self.new_sample, False)

            self.pipeline.add(videodepay)
            self.pipeline.add(queue)
            self.pipeline.add(h264parse)
            self.pipeline.add(decoder)
            self.pipeline.add(videoconvert)
            self.pipeline.add(jpegenc)
            self.pipeline.add(sink)
            videodepay.link(queue)
            queue.link(h264parse)
            h264parse.link(decoder)
            decoder.link(videoconvert)
            videoconvert.link(jpegenc)
            jpegenc.link(sink)
            pad.link(videodepay.get_static_pad("sink"))  # type: ignore[arg-type]

            videodepay.sync_state_with_parent()
            queue.sync_state_with_parent()
            h264parse.sync_state_with_parent()
            videoconvert.sync_state_with_parent()
            decoder.sync_state_with_parent()
            jpegenc.sync_state_with_parent()
            sink.sync_state_with_parent()
        elif pad_name.startswith("audio"):
            print("audio channel ignored")
            fakesink = Gst.ElementFactory.make("fakesink")
            assert fakesink is not None
            self.pipeline.add(fakesink)
            pad.link(fakesink.get_static_pad("sink"))  # type: ignore[arg-type]

            fakesink.sync_state_with_parent()

    def __del__(self) -> None:
        Gst.deinit()

    def get_bus(self) -> Gst.Bus:
        return self.pipeline.get_bus()

    def record(self) -> None:
        # Start playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Error starting playback.")
            exit(-1)

        self._thread_bus_calls = Thread(target=self._handle_bus_calls, daemon=True)
        self._thread_bus_calls.start()
        print("recording ... (ctrl+c to quit)")

    def stop(self) -> None:
        print("stopping")
        self._loop.quit()
        self.pipeline.send_event(Gst.Event.new_eos())
        self.pipeline.set_state(Gst.State.NULL)

    def _handle_bus_calls(self) -> None:
        print("starting bus message loop")
        bus = self.pipeline.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, self.bus_message_cb, self._loop)
        self._loop.run()  # type: ignore[no-untyped-call]
        bus.remove_watch()
        print("bus message loop stopped")

    def bus_message_cb(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            return False

        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print("Error: %s: %s" % (err, debug))
            return False

        return True