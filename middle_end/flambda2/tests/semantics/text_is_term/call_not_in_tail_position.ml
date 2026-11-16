(* A call in tail position that the source marks [@nontail]: it is not made as a
   tail call, and the term says so of the call. The text writes [nontail] in the
   call, so the text is the term. *)

let[@inline never] f x = x

let g x = f x [@nontail]
