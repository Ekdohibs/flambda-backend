module type ORD = sig
  type t

  val compare : t -> t -> int
end

module Make (Ord : ORD) : sig
  val lt : Ord.t -> Ord.t -> bool

  val smaller : Ord.t -> Ord.t -> Ord.t

  val smallest : Ord.t -> Ord.t -> Ord.t -> Ord.t
end
