(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Controls of what the test tool does with the list of code ids of a call of
   known arity (decision D59; rocq/TRUST-LEDGER.md, "The code ids that a call of
   known arity lists"): [Flsemantics.close_known_arity_lists], which extends a
   list under "newer version of", and [Flsemantics.undefined_by_an_open_list],
   which says of a state that has no step whether the unit is to be refused for
   a list. The compiler makes few of the cases, so each is a term of the model
   made here, with the metadata of its code given beside it: [older] is what a
   unit says of the code that it binds with a body, [table] what the table of
   code of the pipeline says. ../semantics/known_arity_lists/lists.expected has
   what this prints. *)

(* The matches below name the calls of known arity and take every other call
   together. *)
[@@@ocaml.warning "-4"]

open Import
module Kinds = Flambda2_rocq_semantics.FlambdaKindModel
module Model = Flambda2_rocq_semantics.FlambdaModel
module Values = Flambda2_rocq_semantics.FlambdaValueSpec
module S = Flsemantics

let unit_of_the_code =
  Compilation_unit.create Compilation_unit.Prefix.empty
    (Compilation_unit.Name.of_string "Known_arity_lists")

let names : (Model.code_id * string) list ref = ref []

let code name =
  let id = Code_id.create ~name ~debug:Debuginfo.none unit_of_the_code in
  names := (S.Code_ids.find id, name) :: !names;
  id

let number = S.Code_ids.find

let name_of id =
  match List.assoc_opt id !names with Some name -> name | None -> "?"

let k = S.positive_of_int 1

let exn = S.positive_of_int 2

let param = S.positive_of_int 1

let my_closure = S.positive_of_int 2

let one_value : Model.arity = [Singleton Kinds.Value]

(* A call of known arity of [callee], with one argument, and with the list given
   or with none. *)
let call ?(callee : Model.simple = Var my_closure)
    ?(args : Model.simple list = [Const (Tagged_immediate (S.z_of_int 1))])
    ?(return_arity : Model.arity = one_value) (listed : Code_id.t list option) :
    Model.expr =
  Apply
    { ap_callee = callee;
      ap_call_kind = Indirect_known_arity (Option.map (List.map number) listed);
      ap_args = args;
      ap_continuation = k;
      ap_exn_continuation = exn;
      ap_exn_extra_args = [];
      ap_args_arity = List.map (fun _ -> List.hd one_value) args;
      ap_return_arity = return_arity
    }

(* A piece of code of one parameter bound with the body given. *)
let code_with (id : Code_id.t) (body : Model.expr) : Model.static_binding =
  Bind_code
    ( number id,
      Code
        ( [param],
          my_closure,
          k,
          exn,
          Model.make_code_metadata one_value one_value,
          body ) )

(* Code that returns its parameter. *)
let returns_its_parameter : Model.expr =
  Apply_cont
    { ac_continuation = k; ac_args = [Var param]; ac_trap_action = None }

(* A group that binds each piece of code, each returning its parameter. *)
let bind (ids : Code_id.t list) (rest : Model.expr) : Model.expr =
  Let_static (List.map (fun id -> code_with id returns_its_parameter) ids, rest)

(* The lists of the calls of known arity of an expression, in order; [None] for
   a call that has no list. *)
let rec lists (e : Model.expr) : Model.code_id list option list =
  let in_handler (Model.Handler (_, _, handler)) = lists handler in
  match e with
  | Let (_, _, _, body) -> lists body
  | Let_static (bindings, body) -> lists_of_bindings bindings @ lists body
  | Let_cont_non_recursive (_, handler, body) -> in_handler handler @ lists body
  | Let_cont_recursive (handlers, body) ->
    List.concat_map (fun (_, handler) -> in_handler handler) handlers
    @ lists body
  | Apply { ap_call_kind = Indirect_known_arity listed; _ } -> [listed]
  | Apply _ | Apply_cont _ | Switch _ | Invalid -> []

and lists_of_bindings (bindings : Model.static_binding list) =
  List.concat_map
    (fun (binding : Model.static_binding) ->
      match binding with
      | Bind_code (_, Code (_, _, _, _, _, body)) -> lists body
      | Bind_deleted_code _ | Bind_symbol _ -> [])
    bindings

(* The metadata of a case. *)
let with_metadata ~(older : (Code_id.t * Code_id.t option) list)
    ~(table : (Code_id.t * Code_id.t option) list) =
  S.code_older
    := List.fold_left
         (fun map (id, older) -> Code_id.Map.add id older map)
         Code_id.Map.empty older;
  S.older_version_in_run
    := fun id ->
         List.find_map
           (fun (id', older) ->
             if Code_id.equal id id' then Some older else None)
           table

(* A list as it is after the extension, and what a chain from each of [others]
   says of it. *)
let print_list where ~(others : Code_id.t list)
    (listed : Model.code_id list option) =
  match listed with
  | None -> Printf.printf "  %s: no list\n" where
  | Some ids ->
    Printf.printf "  %s: [%s]\n" where
      (String.concat "; " (List.map name_of ids));
    List.iter
      (fun other ->
        let n = number other in
        if not (List.mem n ids)
        then
          Printf.printf "    a chain from %s: %s\n" (name_of n)
            (match S.chain_reaches ~numbers:(S.Code_ids.bindings ()) ids n with
            | Some true -> "reaches the list"
            | Some false -> "does not reach the list"
            | None -> "not known"))
      others

let unit body : S.converted =
  { body; return_continuation = k; exn_continuation = exn }

(* The extension of the lists of a prefix and a unit. *)
let extension what ~(older : (Code_id.t * Code_id.t option) list)
    ?(table : (Code_id.t * Code_id.t option) list = [])
    ?(prefix : Model.static_binding list list = []) (after : Model.expr) =
  with_metadata ~older ~table;
  let others = List.map fst older @ List.map fst table in
  let prefix, _, after =
    S.close_known_arity_lists
      { S.no_prefix with prefix_groups = prefix }
      (unit Model.Invalid) (unit after)
  in
  Printf.printf "%s\n" what;
  List.iter
    (print_list "in the prefix" ~others)
    (List.concat_map lists_of_bindings prefix.prefix_groups);
  List.iter (print_list "in the unit" ~others) (lists after.body)

(* A run of a unit, and what the tool says of the state that has no step, if the
   run is undefined. The lists are as they are given, unless [extended]: then
   the lists of the prefix and of the unit are extended first, and the unit is
   run with the prefix in front of it. *)
let run what ~(older : (Code_id.t * Code_id.t option) list)
    ?(table : (Code_id.t * Code_id.t option) list = [])
    ?(prefix : Model.static_binding list list = []) ?(extended = false)
    (body : Model.expr) =
  with_metadata ~older ~table;
  Printf.printf "%s\n" what;
  let prefix = { S.no_prefix with prefix_groups = prefix } in
  let prefix, unit_run =
    if extended
    then
      let prefix, _, after =
        S.close_known_arity_lists prefix (unit Model.Invalid) (unit body)
      in
      prefix, after
    else prefix, unit body
  in
  match
    S.run_to_end ~statics:Values.empty_static_env
      (S.with_prefix prefix unit_run)
  with
  | S.Undefined, _, Some state -> (
    match S.undefined_by_an_open_list state with
    | None -> Printf.printf "  undefined, and reported as of any unit\n"
    | Some (id, why) ->
      Printf.printf "  undefined, and the unit is not checked: %s %s\n"
        (name_of id) why)
  | S.Undefined, _, None -> Printf.printf "  undefined, with no state\n"
  | (S.Returns _ | S.Raises _ | S.Exits_elsewhere | S.Still_running), _, _ ->
    Printf.printf "  not undefined\n"

let () =
  let old = code "old" in
  let newer = code "newer" in
  let middle = code "middle" in
  let newest = code "newest" in
  let root = code "root" in
  let left = code "left" in
  let right = code "right" in
  let round = code "round" in
  let about = code "about" in
  let wrapper = code "wrapper" in
  let far = code "far" in
  Printf.printf "THE EXTENSION OF A LIST\n\n";
  extension "one link, from the newer code to the listed one: added"
    ~older:[old, None; newer, Some old]
    (bind [old; newer] (call (Some [old])));
  extension
    "two links, the middle code declared deleted, its older version in the \
     table: added"
    ~older:[old, None; newest, Some middle]
    ~table:[middle, Some old]
    (Let_static
       ( [Bind_deleted_code (number middle)],
         bind [old; newest] (call (Some [old])) ));
  extension "the same with no table: the middle link is missing, not added"
    ~older:[old, None; newest, Some middle]
    (Let_static
       ( [Bind_deleted_code (number middle)],
         bind [old; newest] (call (Some [old])) ));
  extension
    "the listed code id has no metadata anywhere: the link to it is enough"
    ~older:[newer, Some old]
    (bind [newer] (call (Some [old])));
  extension "the newer code listed: the older is not added"
    ~older:[old, None; newer, Some old]
    (bind [old; newer] (call (Some [newer])));
  extension "two versions of one code, one listed: the other is not added"
    ~older:[root, None; left, Some root; right, Some root]
    (bind [root; left; right] (call (Some [left])));
  extension "a cycle that has no listed code id: nothing added"
    ~older:[old, None; round, Some about; about, Some round]
    (bind [old; round; about] (call (Some [old])));
  extension "a cycle one of whose code ids is listed: the other is added"
    ~older:[round, Some about; about, Some round]
    (bind [round; about] (call (Some [about])));
  extension "one chain known, one with a missing link: one added"
    ~older:[old, None; newer, Some old; newest, Some middle]
    (bind [old; newer; newest] (call (Some [old])));
  extension
    "a call in the code of the prefix, a newer version bound by the unit: not \
     added"
    ~older:[far, Some old]
    ~table:[wrapper, None; old, None]
    ~prefix:
      [ [ code_with wrapper (call (Some [old]));
          code_with old returns_its_parameter ] ]
    (bind [far] Model.Invalid);
  extension
    "a call in the code of the prefix, a newer version in the prefix: added"
    ~older:[]
    ~table:[wrapper, None; old, None; newer, Some old]
    ~prefix:
      [ [ code_with wrapper (call (Some [old]));
          code_with old returns_its_parameter;
          code_with newer returns_its_parameter ] ]
    Model.Invalid;
  extension "a call with no list: left alone"
    ~older:[old, None; newer, Some old]
    (bind [old; newer] (call None));
  extension "the empty list: stays empty"
    ~older:[old, None; newer, Some old]
    (bind [old; newer] (call (Some [])));
  extension
    "the newer version bound by a group after the code that has the call: out \
     of scope, not added"
    ~older:[old, None; wrapper, None; newer, Some old]
    (Let_static
       ( [ code_with wrapper (call (Some [old]));
           code_with old returns_its_parameter ],
         bind [newer] Model.Invalid ));
  extension
    "the newer version bound by the body of a Let_cont, the call in its \
     handler: in scope, added"
    ~older:[old, None; newer, Some old]
    (Let_cont_non_recursive
       ( k,
         Handler ([], Model.make_handler_metadata [], call (Some [old])),
         bind [old; newer] Model.Invalid ));
  extension
    "three listed, one of them twice, and one added: in one order on the \
     numbers of the model, none twice"
    ~older:[old, None; newer, Some old; root, None; left, Some root]
    (bind [old; newer; root; left] (call (Some [left; root; old; left])));
  Printf.printf "\nA RUN THAT STOPS AT A CALL OF KNOWN ARITY\n\n";
  let symbol = S.positive_of_int 1 in
  (* A unit that binds the code, makes a closure of [of_code], and calls it with
     the list given as it is: the extension is not run. *)
  let calling ?args codes ~of_code listed : Model.expr =
    bind codes
      (Let_static
         ( [ Bind_symbol
               (symbol, Static_const_set_of_closures (number of_code, [])) ],
           call ~callee:(Symbol symbol) ?args listed ))
  in
  run "the code of the callee is listed"
    ~older:[old, None]
    (calling [old] ~of_code:old (Some [old]));
  run
    "the code of the callee is a newer version of the listed code and is not \
     in the list"
    ~older:[old, None; newer, Some old]
    (calling [old; newer] ~of_code:newer (Some [old]));
  run "the code of the callee has an older version of which nothing is known"
    ~older:[old, None; newest, Some middle]
    (calling [old; newest] ~of_code:newest (Some [old]));
  run "the same, the table has the older version"
    ~older:[old, None; newest, Some middle]
    ~table:[middle, Some old]
    (calling [old; newest] ~of_code:newest (Some [old]));
  run "the code of the callee is known to be no version of the listed code"
    ~older:[root, None; left, Some root; right, Some root]
    (calling [root; left; right] ~of_code:right (Some [left]));
  run "the empty list, whatever is known of the code of the callee"
    ~older:[old, None; newest, Some middle]
    (calling [old; newest] ~of_code:newest (Some []));
  run
    "a newer version that is not in the list, and two arguments for code of \
     one parameter"
    ~older:[old, None; newer, Some old]
    (calling [old; newer] ~of_code:newer
       ~args:
         [ Const (Tagged_immediate (S.z_of_int 1));
           Const (Tagged_immediate (S.z_of_int 2)) ]
       (Some [old]));
  run "the callee is a number"
    ~older:[old, None; newer, Some old]
    (bind [old; newer]
       (call ~callee:(Const (Tagged_immediate (S.z_of_int 0))) (Some [old])));
  run "the run stops at something that is not a call"
    ~older:[old, None; newer, Some old]
    (bind [old; newer] Model.Invalid);
  run "a call with no list, of the newer version"
    ~older:[old, None; newer, Some old]
    (calling [old; newer] ~of_code:newer None);
  run
    "a newer version that is not in the list, and an argument that has no value"
    ~older:[old, None; newer, Some old]
    (calling [old; newer] ~of_code:newer
       ~args:[Var (S.positive_of_int 99)]
       (Some [old]));
  run
    "a newer version that is not in the list, and a result of another arity \
     than the code returns"
    ~older:[old, None; newer, Some old]
    (bind [old; newer]
       (Let_static
          ( [ Bind_symbol
                (symbol, Static_const_set_of_closures (number newer, [])) ],
            call ~callee:(Symbol symbol) ~return_arity:[] (Some [old]) )));
  (* The code of the prefix calls its parameter with a list; the unit binds a
     newer version of the listed code, makes a closure of it, and gives it to
     that code. The lists are extended first. *)
  let closure_of_wrapper = S.positive_of_int 2 in
  let wrapper_calls_its_parameter =
    code_with wrapper (call ~callee:(Var param) (Some [old]))
  in
  let gives_the_wrapper of_code : Model.expr =
    Let_static
      ( [ Bind_symbol (symbol, Static_const_set_of_closures (number of_code, []));
          Bind_symbol
            ( closure_of_wrapper,
              Static_const_set_of_closures (number wrapper, []) ) ],
        Apply
          { ap_callee = Symbol closure_of_wrapper;
            ap_call_kind = Direct (number wrapper);
            ap_args = [Symbol symbol];
            ap_continuation = k;
            ap_exn_continuation = exn;
            ap_exn_extra_args = [];
            ap_args_arity = one_value;
            ap_return_arity = one_value
          } )
  in
  run
    "code of the prefix calls its parameter with a list; the unit gives it a \
     closure of a newer version that the unit binds (the lists extended first)"
    ~older:[far, Some old]
    ~table:[wrapper, None; old, None]
    ~prefix:[[wrapper_calls_its_parameter; code_with old returns_its_parameter]]
    ~extended:true
    (bind [far] (gives_the_wrapper far));
  run "the same, the unit gives it a closure of the listed code"
    ~older:[far, Some old]
    ~table:[wrapper, None; old, None]
    ~prefix:[[wrapper_calls_its_parameter; code_with old returns_its_parameter]]
    ~extended:true
    (bind [far] (gives_the_wrapper old));
  run
    "the same, the newer version is code of the prefix: it is in the list once \
     extended"
    ~older:[]
    ~table:[wrapper, None; old, None; newer, Some old]
    ~prefix:
      [ [ wrapper_calls_its_parameter;
          code_with old returns_its_parameter;
          code_with newer returns_its_parameter ] ]
    ~extended:true (gives_the_wrapper newer)
