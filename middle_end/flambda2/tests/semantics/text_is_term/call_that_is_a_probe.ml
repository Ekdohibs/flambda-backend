(* Two probes: the call of the handler of each is made only while the probe is
   enabled, and the second is enabled from the start. The text says of each call
   that it is a probe, with its name ([probe("p")] and [probe("q"
   enabled_at_init)]); it said nothing, and the call was read as a plain call,
   which is always made. So the text is the term. *)

external ignore : 'a -> unit = "%ignore"

let f (x : int) =
  [%probe "p" (ignore x)];
  [%probe "q" ~enabled_at_init:true (ignore x)];
  x
