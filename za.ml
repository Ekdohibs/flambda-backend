let f x =
  let[@zero_alloc] g x = [x] in
  g x