let of_message message =
  let length = String.length message in
  let start = if length > 0 && Char.equal message.[0] '(' then 1 else 0 in
  let is_in_name c =
    match c with
    | 'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' | '_' | '\'' -> true
    | _ -> false
  in
  let rec end_of_name i =
    if i < length && is_in_name message.[i] then end_of_name (i + 1) else i
  in
  let stop = end_of_name start in
  let begins_with_capital =
    stop > start && match message.[start] with 'A' .. 'Z' -> true | _ -> false
  in
  let ends_the_name =
    stop = length
    || start = 1
       &&
       match message.[stop] with
       | ' ' | '\n' | '\t' | ')' -> true
       | _ -> false
  in
  if begins_with_capital && ends_the_name
  then Some (String.sub message start (stop - start))
  else None

let is_name name =
  match of_message name with
  | Some name' -> String.equal name name'
  | None -> false
