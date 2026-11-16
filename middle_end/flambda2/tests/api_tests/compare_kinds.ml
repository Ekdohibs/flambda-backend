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

   Each pair of kinds below is put in eight selected places of a term where the
   comparison reads a kind with its subkind, and the answer is printed. The
   places that are not here are the arities of the parameters and of the result
   of code, which are compared by the function that compares the arities of a
   call (compare_texts has pairs for them, and ../semantics/text_is_term/ has
   units of the two orders), and the kinds of the invariant parameters of
   recursive continuations (compare_texts).

   Then pairs that differ in what stands beside a kind (the grouping of an
   arity; the tag, the mutability and the allocation mode of a block that is
   made; the mutability of a read and the kind of a write at an offset), and
   things that a text cannot write: that a call is a probe, a call that performs
   an effect, the shape of a static block, the debugging information of its
   fields (which is not compared, as no other debugging information is), and a
   group of two recursive continuations. *)

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

let call ?(probe = None) ?(callee = Some argument)
    ?(call_kind = Call_kind.indirect_function_call_unknown_arity) ~args_arity
    ~return_arity ~extra_args () =
  unit (fun ~return_continuation ~exn_continuation ~toplevel_my_alloc_region ->
      Flambda.Apply.create ~callee ~continuation:(Return return_continuation)
        (Exn_continuation.create ~exn_handler:exn_continuation ~extra_args)
        ~args:(List.map (fun _ -> argument) (Flambda_arity.unarize args_arity))
        ~args_arity ~return_arity ~call_kind
        ~return_mode:
          (Alloc_mode.For_applications.not_alloc_stack
             ~alloc_region:toplevel_my_alloc_region)
        Debuginfo.none ~inlined:Default_inlined
        ~inlining_state:(Inlining_state.default ~round:0)
        ~probe ~position:Normal
        ~relative_history:Inlining_history.Relative.empty
      |> Flambda.Expr.create_apply)

let jump continuation args =
  Flambda.Apply_cont.create continuation ~args ~dbg:Debuginfo.none
  |> Flambda.Expr.create_apply_cont

(* A unit that binds a variable to the primitive and returns it. *)
let let_prim prim =
  unit
    (fun ~return_continuation ~exn_continuation:_ ~toplevel_my_alloc_region ->
      let result = Variable.create "result" K.value in
      Flambda.Let.create
        (Bound_pattern.singleton
           (Bound_var.create result Flambda_debug_uid.none Name_mode.normal))
        (Flambda.Named.create_prim
           (prim ~toplevel_my_alloc_region)
           Debuginfo.none)
        ~body:(jump return_continuation [Simple.var result])
        ~free_names_of_body:Unknown
      |> Flambda.Expr.create_let)

let one kind = Flambda_arity.create_singletons [kind]

let any = one KS.any_value

let places =
  [ ( "the arity of the arguments of a call",
      fun kind ->
        call ~args_arity:(one kind) ~return_arity:any ~extra_args:[] () );
    ( "the arity of the arguments of a call, in an unboxed product",
      fun kind ->
        call
          ~args_arity:
            (Flambda_arity.create
               [Unboxed_product [Singleton KS.naked_float; Singleton kind]])
          ~return_arity:any ~extra_args:[] () );
    ( "the arity of the result of a call",
      fun kind ->
        call ~args_arity:any ~return_arity:(one kind) ~extra_args:[] () );
    ( "the kind of an extra argument of an exception continuation",
      fun kind ->
        call ~args_arity:any ~return_arity:any ~extra_args:[argument, kind] ()
    );
    ( "the kinds of the fields of a block that is made",
      fun kind ->
        let_prim (fun ~toplevel_my_alloc_region ->
            Variadic
              ( Make_block
                  ( Values (Tag.Scannable.zero, [KS.any_value; kind]),
                    Immutable,
                    Alloc_mode.For_allocations.heap
                      ~alloc_region:toplevel_my_alloc_region ),
                [argument; argument] )) );
    ( "the kind of a read at an offset",
      fun kind ->
        let_prim (fun ~toplevel_my_alloc_region:_ ->
            Binary (Read_offset (kind, Immutable), argument, argument)) );
    ( "the kind of a write at an offset",
      fun kind ->
        let_prim (fun ~toplevel_my_alloc_region:_ ->
            Ternary
              ( Write_offset (Into_block, kind, Alloc_mode.For_assignments.heap),
                argument,
                argument,
                argument )) );
    ( "the kind of a parameter of a continuation",
      fun kind ->
        unit
          (fun
            ~return_continuation
            ~exn_continuation:_
            ~toplevel_my_alloc_region:_
          ->
            let continuation = Continuation.create () in
            let param = Variable.create "param" K.value in
            let handler =
              Flambda.Continuation_handler.create
                (Bound_parameters.create
                   [Bound_parameter.create param kind Flambda_debug_uid.none])
                ~handler:(jump return_continuation [Simple.var param])
                ~free_names_of_handler:Unknown ~is_exn_handler:false
                ~is_cold:false
            in
            Flambda.Let_cont.create_non_recursive continuation handler
              ~body:(jump continuation [argument])
              ~free_names_of_body:Unknown) ) ]

let answer unit1 unit2 =
  match Flambda2_compare.Compare.flambda_units unit1 unit2 with
  | Equivalent -> "the same"
  | Different { approximant = _ } -> "different"
  | exception exn -> "STOPPED: " ^ Printexc.to_string exn

let () =
  List.iter
    (fun (what, kind1, kind2) ->
      Printf.printf "%s (the kinds, by Flambda_kind.With_subkind.equal: %s)\n"
        what
        (if KS.equal kind1 kind2 then "equal" else "not equal");
      List.iter
        (fun (place, make) ->
          Printf.printf "  %s: %s\n" place (answer (make kind1) (make kind2)))
        places)
    pairs

let probe name = Probe.from_lambda (Some { name; enabled_at_init = false })

let call_that_is probe =
  call ~probe ~args_arity:any ~return_arity:any ~extra_args:[] ()

(* Debugging information that is not empty. *)
let somewhere =
  let position : Lexing.position =
    { pos_fname = "kinds.ml"; pos_lnum = 1; pos_bol = 0; pos_cnum = 0 }
  in
  Debuginfo.from_location
    (Debuginfo.Scoped_location.of_location
       ~scopes:Debuginfo.Scoped_location.empty_scopes
       { loc_start = position;
         loc_end = { position with pos_cnum = 1 };
         loc_ghost = false
       })

(* The operation of an effect has no callee and no arguments: what it is given
   is in the kind of the call. *)
let call_that_performs eff =
  call ~callee:None
    ~call_kind:(Call_kind.effect_ (Call_kind.Effect.perform ~eff))
    ~args_arity:Flambda_arity.nullary ~return_arity:any ~extra_args:[] ()

let static_block ?(dbg = Debuginfo.none) shape =
  unit
    (fun ~return_continuation ~exn_continuation:_ ~toplevel_my_alloc_region:_ ->
      let block =
        Symbol.create compilation_unit (Linkage_name.of_string "block")
      in
      let body =
        Flambda.Apply_cont.create return_continuation
          ~args:[Simple.symbol block]
          ~dbg:Debuginfo.none
        |> Flambda.Expr.create_apply_cont
      in
      Flambda.Let.create
        (Bound_pattern.static
           (Bound_static.singleton (Bound_static.Pattern.block_like block)))
        (Flambda.Named.create_static_consts
           (Flambda.Static_const_group.create
              [ Flambda.Static_const_or_code.create_static_const
                  (Static_const.block Tag.Scannable.zero Immutable shape
                     [ Simple.With_debuginfo.create argument dbg;
                       Simple.With_debuginfo.create argument dbg ]) ]))
        ~body ~free_names_of_body:Unknown
      |> Flambda.Expr.create_let)

(* A group of [n] recursive continuations, of which the first is called. *)
let recursive_group n =
  unit
    (fun ~return_continuation ~exn_continuation:_ ~toplevel_my_alloc_region:_ ->
      let continuations = List.init n (fun _ -> Continuation.create ()) in
      let handler =
        Flambda.Continuation_handler.create Bound_parameters.empty
          ~handler:(jump return_continuation [argument])
          ~free_names_of_handler:Unknown ~is_exn_handler:false ~is_cold:false
      in
      Flambda.Let_cont.create_recursive
        (Continuation.Lmap.of_list
           (List.map (fun continuation -> continuation, handler) continuations))
        ~invariant_params:Bound_parameters.empty
        ~body:(jump (List.hd continuations) []))

(* The other fields of the primitives whose kinds are compared here. *)
let block ?(tag = Tag.Scannable.zero) ?(mutability = Mutability.Immutable)
    ?(local = false) () =
  let_prim (fun ~toplevel_my_alloc_region ->
      Variadic
        ( Make_block
            ( Values (tag, [reference]),
              mutability,
              if local
              then
                Alloc_mode.For_allocations.local
                  ~alloc_region:toplevel_my_alloc_region
                  ~region:toplevel_my_alloc_region
              else
                Alloc_mode.For_allocations.heap
                  ~alloc_region:toplevel_my_alloc_region ),
          [argument] ))

let read flag =
  let_prim (fun ~toplevel_my_alloc_region:_ ->
      Binary (Read_offset (reference, flag), argument, argument))

let write write_kind =
  let_prim (fun ~toplevel_my_alloc_region:_ ->
      Ternary
        ( Write_offset (write_kind, reference, Alloc_mode.For_assignments.heap),
          argument,
          argument,
          argument ))

let call_with_arguments args_arity =
  call ~args_arity ~return_arity:any ~extra_args:[] ()

let () =
  let float = Flambda_arity.Component_for_creation.Singleton KS.naked_float in
  let value = Flambda_arity.Component_for_creation.Singleton KS.any_value in
  let arity = Flambda_arity.create in
  Printf.printf "two arities that group the same kinds in the same way: %s\n"
    (answer
       (call_with_arguments (arity [Unboxed_product [float; value]; value]))
       (call_with_arguments (arity [Unboxed_product [float; value]; value])));
  Printf.printf "a product of two kinds and the two kinds: %s\n"
    (answer
       (call_with_arguments (arity [Unboxed_product [float; value]]))
       (call_with_arguments (arity [float; value])));
  Printf.printf "a product of three kinds, nested in two ways: %s\n"
    (answer
       (call_with_arguments
          (arity [Unboxed_product [Unboxed_product [float; value]; value]]))
       (call_with_arguments
          (arity [Unboxed_product [float; Unboxed_product [value; value]]])));
  Printf.printf "an arity with the product of nothing and one without: %s\n"
    (answer
       (call_with_arguments (arity [value; Unboxed_product []]))
       (call_with_arguments (arity [value])));
  Printf.printf "two blocks made alike: %s\n" (answer (block ()) (block ()));
  Printf.printf "two blocks made with two tags: %s\n"
    (answer (block ()) (block ~tag:(tag 1) ()));
  Printf.printf "a block made immutable and one made mutable: %s\n"
    (answer (block ()) (block ~mutability:Mutable ()));
  Printf.printf "a block made on the heap and one made on the stack: %s\n"
    (answer (block ()) (block ~local:true ()));
  Printf.printf "two reads at an offset, of an immutable and of a mutable: %s\n"
    (answer (read Immutable) (read Mutable));
  Printf.printf "two writes at an offset, of two kinds of write: %s\n"
    (answer (write Into_block) (write Into_block_or_off_heap))

let () =
  Printf.printf "a call that is a probe and a plain call: %s\n"
    (answer (call_that_is (probe "a")) (call_that_is None));
  Printf.printf "two calls that are probes of one name: %s\n"
    (answer (call_that_is (probe "a")) (call_that_is (probe "a")));
  Printf.printf "two calls that are probes of two names: %s\n"
    (answer (call_that_is (probe "a")) (call_that_is (probe "b")));
  Printf.printf "two calls that perform one effect: %s\n"
    (answer (call_that_performs argument) (call_that_performs argument));
  Printf.printf "two calls that perform two effects: %s\n"
    (answer
       (call_that_performs argument)
       (call_that_performs (Simple.const_zero machine_width)));
  let mixed : K.Scannable_block_shape.t =
    Mixed_record
      (K.Mixed_block_shape.from_prefix_size_and_suffix_elements 1 [Naked_float])
  in
  Printf.printf "two static blocks of values: %s\n"
    (answer (static_block Value_only) (static_block Value_only));
  Printf.printf "a static block of values and a mixed one: %s\n"
    (answer (static_block Value_only) (static_block mixed));
  Printf.printf
    "two static blocks whose fields have not the same debugging information \
     (one has some: %b): %s\n"
    (not (Debuginfo.is_none somewhere))
    (answer (static_block Value_only) (static_block ~dbg:somewhere Value_only));
  Printf.printf "two groups of two recursive continuations: %s\n"
    (answer (recursive_group 2) (recursive_group 2));
  Printf.printf "a group of one recursive continuation and a group of two: %s\n"
    (answer (recursive_group 1) (recursive_group 2))
