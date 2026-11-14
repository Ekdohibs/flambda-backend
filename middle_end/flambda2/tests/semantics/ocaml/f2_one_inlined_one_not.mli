module type Shift = sig
  val by : int
end

module Add (S : Shift) : sig
  val apply : int -> int

  val twice : int -> int
end

module Add_three : sig
  val apply : int -> int

  val twice : int -> int
end

module Add_five : sig
  val apply : int -> int

  val twice : int -> int
end

val result : int
