(* Reads, writes and lengths of arrays of unboxed products: of a product of two,
   and of products that hold a product, which stands after a component that is
   not one in the first and before it in the second. The text writes the kind
   of such an array as a list in brackets, with the product inside it as a
   list again, and the reader reads the list in the order of the text. It
   found a flag before a list wherever they stood: it took the kind of what a
   read gives for the kind of the array, and it read a product that stands
   before a component that is not one after it. So the text is the term. *)

external get : ('a : any mod separable). 'a array -> int -> 'a
  = "%array_safe_get"
[@@layout_poly]

external set : ('a : any mod separable). 'a array -> int -> 'a -> unit
  = "%array_safe_set"
[@@layout_poly]

external length : ('a : any mod separable). 'a array -> int = "%array_length"
[@@layout_poly]

let get_pair (a : #(float# * int) array) i = get a i

let set_pair (a : #(float# * int) array) i x = set a i x

let get_after (a : #(int * #(float# * int)) array) i = get a i

let set_after (a : #(int * #(float# * int)) array) i x = set a i x

let get_before (a : #(#(float# * int) * int) array) i = get a i

let set_before (a : #(#(float# * int) * int) array) i x = set a i x

let length_after (a : #(int * #(float# * int)) array) = length a

let length_before (a : #(#(float# * int) * int) array) = length a
