(* Seventeen functions in the body of a functor, each calling the one before,
   none using the argument of the functor. Simplify makes the seventeen closures
   and the block of the result of the functor symbols. A lifting of the
   validation takes only a value whose parts are symbols or constants already,
   so the validation would lift them one after the other, eighteen liftings in
   turn, which is more than the test tool tries: the unit is here for the line
   that says so. *)

external ( + ) : int -> int -> int = "%addint"

module Make (X : sig end) = struct
  let a x = x + 1

  let b x = a x

  let c x = b x

  let d x = c x

  let e x = d x

  let f x = e x

  let g x = f x

  let h x = g x

  let i x = h x

  let j x = i x

  let k x = j x

  let l x = k x

  let m x = l x

  let n x = m x

  let o x = n x

  let p x = o x

  let q x = p x
end
