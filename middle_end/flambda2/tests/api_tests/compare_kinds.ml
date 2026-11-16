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

(* What [Compare.flambda_units] says of two units that differ only in one kind
   with a subkind.

   The kind of a variant holds the set of its constant constructors and the map
   of the others. [Flambda_kind.With_subkind.equal] compares them as balanced
   trees, and two trees that hold the same elements differ if the elements were
   added in two orders. The comparison of units must call two such kinds the
   same, and must still tell apart two kinds that hold different things.

   Each pair of kinds below is put in each place of a term where the comparison
   reads a kind with its subkind, and the answer is printed. *)

open Flambda2_bound_identifiers
open Flambda2_identifiers
open Flambda2_kinds
open Flambda2_nominal
open Flambda2_numbers
open Flambda2_term_basics
open Flambda2_terms
module K = Flambda_kind
module KS = Flambda_kind.With_subkind

let machine_width = Target_system.Machine_width.Sixty_four

let const i = Target_ocaml_int.of_int machine_width i

let tag i = Tag.Scannable.create_exn i

let fields kinds : KS.Non_null_value_subkind.constructor_shape =
  Determined (K.Block_shape.Scannable Value_only, kinds)

(* The elements are added one by one, in the order of the lists. *)
let variant ?(nullable = KS.Nullable.Non_nullable) ~consts ~non_consts () =
  let consts =
    List.fold_left
      (fun set i -> Target_ocaml_int.Set.add (const i) set)
      Target_ocaml_int.Set.empty consts
  in
  let non_consts =
    List.fold_left
      (fun map (i, shape) -> Tag.Scannable.Map.add (tag i) shape map)
      Tag.Scannable.Map.empty non_consts
  in
  KS.create K.value (Variant { consts; non_consts }) nullable

let block0 = 0, fields [KS.tagged_immediate]

let block1 = 1, fields [KS.any_value; KS.any_value]

let reference = variant ~consts:[0; 1] ~non_consts:[block0; block1] ()

(* A variant among the fields of a constructor, itself built in two orders. *)
let nested inner = variant ~consts:[] ~non_consts:[0, fields [inner]] ()

let pairs =
  [ ( "the same variant, its constructors added in the other order",
      reference,
      variant ~consts:[1; 0] ~non_consts:[block1; block0] () );
    ( "the same, inside the field of a constructor",
      nested reference,
      nested (variant ~consts:[1; 0] ~non_consts:[block1; block0] ()) );
    ( "one constant constructor more",
      reference,
      variant ~consts:[0; 1; 2] ~non_consts:[block0; block1] () );
    ( "one constant constructor fewer",
      reference,
      variant ~consts:[0] ~non_consts:[block0; block1] () );
    ( "another kind of field",
      reference,
      variant ~consts:[0; 1]
        ~non_consts:[block0; 1, fields [KS.any_value; KS.tagged_immediate]]
        () );
    ( "one field more",
      reference,
      variant ~consts:[0; 1]
        ~non_consts:
          [block0; 1, fields [KS.any_value; KS.any_value; KS.any_value]]
        () );
    ( "another tag",
      reference,
      variant ~consts:[0; 1] ~non_consts:[block0; 2, snd block1] () );
    ( "one constructor with fields more",
      reference,
      variant ~consts:[0; 1]
        ~non_consts:[block0; block1; 2, fields [KS.any_value]]
        () );
    ( "the fields of a constructor not known",
      reference,
      variant ~consts:[0; 1] ~non_consts:[block0; 1, Undetermined] () );
    ( "another field, inside the field of a constructor",
      nested reference,
      nested (variant ~consts:[0; 1] ~non_consts:[block0; 1, snd block0] ()) );
    ( "may be null",
      reference,
      variant ~nullable:Nullable ~consts:[0; 1] ~non_consts:[block0; block1] ()
    );
    "not a variant", reference, KS.any_value ]

let compilation_unit =
  Compilation_unit.create Compilation_unit.Prefix.empty
    (Compilation_unit.Name.of_string "Kinds")

let () =
  Env.set_current_unit
    (Unit_info.make_dummy ~input_name:"Kinds" compilation_unit)

let module_symbol =
  Flambda2_import.Symbol.for_compilation_unit compilation_unit
  |> Symbol.create_wrapped

let argument = Simple.symbol module_symbol

(* A unit whose body is made by [body] from its return continuation, its
   exception continuation and its region. *)
let unit body =
  let return_continuation = Continuation.create ~sort:Toplevel_return () in
  let exn_continuation = Continuation.create () in
  let toplevel_my_alloc_region = Variable.create "region" K.region in
  Flambda_unit.create ~return_continuation ~exn_continuation
    ~toplevel_my_alloc_region
    ~body:
      (body ~return_continuation ~exn_continuation ~toplevel_my_alloc_region)
    ~module_symbol

let call ~args_arity ~return_arity ~extra_args =
  unit (fun ~return_continuation ~exn_continuation ~toplevel_my_alloc_region ->
      Flambda.Apply.create ~callee:(Some argument)
        ~continuation:(Return return_continuation)
        (Exn_continuation.create ~exn_handler:exn_continuation ~extra_args)
        ~args:(List.map (fun _ -> argument) (Flambda_arity.unarize args_arity))
        ~args_arity ~return_arity
        ~call_kind:Call_kind.indirect_function_call_unknown_arity
        ~return_mode:
          (Alloc_mode.For_applications.not_alloc_stack
             ~alloc_region:toplevel_my_alloc_region)
        Debuginfo.none ~inlined:Default_inlined
        ~inlining_state:(Inlining_state.default ~round:0)
        ~probe:None ~position:Normal
        ~relative_history:Inlining_history.Relative.empty
      |> Flambda.Expr.create_apply)

let one kind = Flambda_arity.create_singletons [kind]

let any = one KS.any_value

let places =
  [ ( "the arity of the arguments of a call",
      fun kind -> call ~args_arity:(one kind) ~return_arity:any ~extra_args:[]
    );
    ( "the arity of the arguments of a call, in an unboxed product",
      fun kind ->
        call
          ~args_arity:
            (Flambda_arity.create
               [Unboxed_product [Singleton KS.naked_float; Singleton kind]])
          ~return_arity:any ~extra_args:[] );
    ( "the arity of the result of a call",
      fun kind -> call ~args_arity:any ~return_arity:(one kind) ~extra_args:[]
    );
    ( "the kind of an extra argument of an exception continuation",
      fun kind ->
        call ~args_arity:any ~return_arity:any ~extra_args:[argument, kind] );
    ( "the kinds of the fields of a block that is made",
      fun kind ->
        unit
          (fun
            ~return_continuation
            ~exn_continuation:_
            ~toplevel_my_alloc_region
          ->
            let block = Variable.create "block" K.value in
            let body =
              Flambda.Apply_cont.create return_continuation
                ~args:[Simple.var block]
                ~dbg:Debuginfo.none
              |> Flambda.Expr.create_apply_cont
            in
            Flambda.Let.create
              (Bound_pattern.singleton
                 (Bound_var.create block Flambda_debug_uid.none Name_mode.normal))
              (Flambda.Named.create_prim
                 (Variadic
                    ( Make_block
                        ( Values (Tag.Scannable.zero, [KS.any_value; kind]),
                          Immutable,
                          Alloc_mode.For_allocations.heap
                            ~alloc_region:toplevel_my_alloc_region ),
                      [argument; argument] ))
                 Debuginfo.none)
              ~body ~free_names_of_body:Unknown
            |> Flambda.Expr.create_let) ) ]

let () =
  List.iter
    (fun (what, kind1, kind2) ->
      Printf.printf "%s (the kinds, by Flambda_kind.With_subkind.equal: %s)\n"
        what
        (if KS.equal kind1 kind2 then "equal" else "not equal");
      List.iter
        (fun (place, make) ->
          let answer =
            match
              Flambda2_compare.Compare.flambda_units (make kind1) (make kind2)
            with
            | Equivalent -> "the same"
            | Different { approximant = _ } -> "different"
            | exception exn -> "STOPPED: " ^ Printexc.to_string exn
          in
          Printf.printf "  %s: %s\n" place answer)
        places)
    pairs
