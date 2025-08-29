open Flambda2_bound_identifiers
open Flambda2_identifiers
open Flambda2_kinds
open Flambda2_nominal
open Flambda2_numbers
open Flambda2_term_basics
module K = Flambda_kind
module T = Flambda2_types
module TE = T.Typing_env

let create_env () =
  let resolver _ = None in
  let get_imported_names () = Name.Set.empty in
  TE.create ~resolver ~get_imported_names

module Evenodd = struct
  module T0 = struct
    type t =
      | Even
      | Odd
      | Unused

    let print ppf = function
      | Even -> Format.fprintf ppf "even"
      | Odd -> Format.fprintf ppf "odd"
      | Unused -> Format.fprintf ppf "unused"

    let compare = Stdlib.compare

    let equal = Stdlib.( = )

    let hash : t -> int = Stdlib.Hashtbl.hash
  end

  include T0
  include Flambda2_algorithms.Container_types.Make (T0)
end

module Rewriter = Flambda2_types.Rewriter
module Rule = Rewriter.Rule

module Traverse = Rewriter.Make (struct
  include Evenodd

  let rewrite t env ty : _ Rule.t =
    match t with
    | Unused ->
      Rule.rewrite Rewriter.Pattern.any
        (Rewriter.Expr.unknown (Flambda2_types.kind ty))
    | Even | Odd -> (
      match Flambda2_types.prove_is_a_tagged_immediate env ty with
      | Proved () ->
        let var = Rewriter.Var.create () in
        Rule.rewrite
          (Rewriter.Pattern.var var Unused)
          (Rewriter.Expr.immutable_block ~is_unique:false Tag.zero
             ~shape:(K.Block_shape.Scannable Value_only)
             (Alloc_mode.For_types.unknown ())
             ~fields:[var])
      | Unknown -> Rule.identity t)

  let block_slot ?tag:_ t field env ty =
    let field = Target_ocaml_int.to_int field in
    if match t with
       | Even -> field mod 2 = 0
       | Odd -> field mod 2 = 1
       | Unused -> false
    then
      match Flambda2_types.prove_is_a_tagged_immediate env ty with
      | Proved () -> Even
      | Unknown -> t
    else Unused

  let array_slot _ _ _ _ = Unused

  let value_slot _ _ _ _ = Unused

  let function_slot _ _ _ _ = Unused
end)

let ( let$ ) (name, ty) k env =
  let kind = T.kind ty in
  let var = Variable.create name kind in
  let bound_var =
    Bound_var.create var Flambda_debug_uid.none Name_mode.normal
  in
  let env = TE.add_definition env (Bound_name.create_var bound_var) kind in
  k var (TE.add_equation env (Name.var var) ty)

let alias var = T.alias_type_of (Variable.kind var) (Simple.var var)

let block tag fields =
  T.immutable_block ~is_unique:false (Tag.create_exn tag)
    ~shape:(K.Block_shape.Scannable Value_only) Alloc_mode.For_types.heap
    ~fields

let immediate = T.any_tagged_immediate

let var name ty = name, ty

let run env k = k env

let return v env = v, env

let test_meet_chains_two_vars () =
  let live_vars, env =
    run (create_env ())
      (let$ imm0 = var "imm0" immediate in
       let$ imm2 = var "imm2" immediate in
       let$ var0 = var "var0" (block 2 [alias imm0; alias imm0]) in
       let$ var1 =
         var "var1" (block 0 [block 1 [alias var0; alias imm2]; alias var0])
       in
       let$ var2 = var "var2" (alias var1) in
       return
         [ Name.var var1, (Evenodd.Even, K.value);
           Name.var var2, (Evenodd.Odd, K.value) ])
  in
  Format.eprintf "Initial situation:@ %a\n%!" TE.print env;
  let rewritten = Traverse.rewrite env (Name.Map.of_list live_vars) in
  Format.eprintf "After rewrite keeping only even/odd fields@ ";
  Format.eprintf "(with live vars @[<h>%a@]):@ %a\n%!"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ")
       Name.print)
    (List.map fst live_vars) TE.print rewritten

let () =
  let comp_unit = "Rewrite_test" |> Compilation_unit.of_string in
  let unit_info = Unit_info.make_dummy ~input_name:"rewrite_test" comp_unit in
  Env.set_unit_name (Some unit_info);
  Format.eprintf "MEET CHAINS WITH TWO VARS@\n@.";
  test_meet_chains_two_vars ()
