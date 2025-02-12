type ('a, 'b) t
val inject : ('a, 'b) t -> 'a -> 'b

val id : ('a, 'a) t

val compose : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t

module Make_covariant (X : sig type +'a t end) : sig
  val fmap : ('a, 'b) t -> ('a X.t, 'b X.t) t
end