external ( + ) : int -> int -> int = "%addint"

external ( - ) : int -> int -> int = "%subint"

external ( <= ) : int -> int -> bool = "%lessequal"

external opaque : 'a -> 'a = "%opaque"

val count_from : int -> int

val twice : int -> int

val result : int
