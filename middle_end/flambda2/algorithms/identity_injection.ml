type ('a, 'b) t = Inject : ('a, 'a) t

let[@inline] inject : type a b. (a, b) t -> a -> b = fun eq x ->
  match eq with Inject -> x

let id = Inject

let[@inline] compose : type a b c. (a, b) t -> (b, c) t -> (a, c) t = fun Inject Inject -> Inject

module Make_covariant (X : sig type +'a t end) = struct
  let fmap : type a b. (a, b) t -> (a X.t, b X.t) t = fun Inject -> Inject
end 