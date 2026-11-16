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
   group of two recursive continuations.

   Then the unit that the comparison gives for two different units, where the
   kind of a call of the first unit has a name that the second unit binds under
   another name.

   Last, invalid expressions, which are compared by their kind: a name read from
   the message that the term keeps, which for a constructor of
   [Flambda.Invalid.t] other than [Message] is the name of the constructor. A
   value of each of those sixteen constructors is built, and four of [Message]:
   of a sentence, of nothing, of the message of another constructor, and of a
   name that no constructor has. For each of the twenty the kind read from its
   message is printed, with a word if the function of the parser that reads the
   kind for the text of a term reads another; and, for the sixteen, with a word
   if it is not the name that the comparison has for the constructor. Then pairs
   of invalid expressions are compared. Then a unit that is one invalid
   expression is written by the printer of texts and read back by the reader,
   for each of these names: the sixteen, [Message], a name that no constructor
   has, and the fifteen words that begin with a capital and are keywords of the
   text, which the printer must write in backquotes. The one argument of the
   test is a text with code, for the constructor that takes the metadata of
   code. *)

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

(* The unit that the comparison gives for two different units is, by
   compare.mli, the first unit with the names of the second where the two agree.
   So the comparison must say of the first unit and of the unit given that they
   are the same. *)
let answer_for_the_unit_given unit1 unit2 =
  match Flambda2_compare.Compare.flambda_units unit1 unit2 with
  | Equivalent -> "NOT DIFFERENT"
  | Different { approximant } -> answer unit1 approximant
  | exception exn -> "STOPPED: " ^ Printexc.to_string exn

let fresh_symbol =
  let count = ref 0 in
  fun () ->
    incr count;
    Symbol.create compilation_unit
      (Linkage_name.of_string (Printf.sprintf "bound%d" !count))

(* A unit that binds a code ID (to deleted code, which is enough for the name)
   and a symbol, each new at every use, and then either jumps or makes one call
   whose kind [call_kind] makes from the two names. *)
let with_names ?call_kind () =
  unit (fun ~return_continuation ~exn_continuation ~toplevel_my_alloc_region ->
      let code_id =
        Code_id.create ~name:"code" ~debug:Debuginfo.none compilation_unit
      in
      let symbol = fresh_symbol () in
      let body =
        match call_kind with
        | None -> jump return_continuation [argument]
        | Some call_kind ->
          let call_kind : Call_kind.t = call_kind ~code_id ~symbol in
          let callee, args, args_arity =
            match call_kind with
            | Effect _ -> None, [], Flambda_arity.nullary
            | Function _ | Method _ | C_call _ -> Some argument, [argument], any
          in
          Flambda.Apply.create ~callee
            ~continuation:(Return return_continuation)
            (Exn_continuation.create ~exn_handler:exn_continuation
               ~extra_args:[])
            ~args ~args_arity ~return_arity:any ~call_kind
            ~return_mode:
              (Alloc_mode.For_applications.not_alloc_stack
                 ~alloc_region:toplevel_my_alloc_region)
            Debuginfo.none ~inlined:Default_inlined
            ~inlining_state:(Inlining_state.default ~round:0)
            ~probe:None ~position:Normal
            ~relative_history:Inlining_history.Relative.empty
          |> Flambda.Expr.create_apply
      in
      Flambda.Let.create
        (Bound_pattern.static
           (Bound_static.create
              [ Bound_static.Pattern.code code_id;
                Bound_static.Pattern.block_like symbol ]))
        (Flambda.Named.create_static_consts
           (Flambda.Static_const_group.create
              [ Flambda.Static_const_or_code.deleted_code;
                Flambda.Static_const_or_code.create_static_const
                  (Static_const.block Tag.Scannable.zero Immutable Value_only
                     [Simple.With_debuginfo.create argument Debuginfo.none]) ]))
        ~body ~free_names_of_body:Unknown
      |> Flambda.Expr.create_let)

let () =
  let direct ~code_id ~symbol:_ = Call_kind.direct_function_call code_id in
  let known_arity ~code_id ~symbol:_ =
    Call_kind.indirect_function_call_known_arity
      ~code_ids:(Known (Code_id.Set.singleton code_id))
  in
  let method_call ~code_id:_ ~symbol =
    Call_kind.method_call Public ~obj:(Simple.symbol symbol)
  in
  let effect_ ~code_id:_ ~symbol =
    Call_kind.effect_ (Call_kind.Effect.perform ~eff:(Simple.symbol symbol))
  in
  List.iter
    (fun (what, call_kind) ->
      Printf.printf "%s:\n  two units that have it: %s\n" what
        (answer (with_names ~call_kind ()) (with_names ~call_kind ()));
      Printf.printf
        "  a unit that has it and one that jumps, the first and the unit \
         given: %s\n"
        (answer_for_the_unit_given (with_names ~call_kind ()) (with_names ())))
    [ "a direct call of the code that the unit binds", direct;
      ( "a call of known arity that may reach the code that the unit binds",
        known_arity );
      "a method call on the symbol that the unit binds", method_call;
      "an effect performed on the symbol that the unit binds", effect_ ]

(* The message that an invalid expression keeps of what it was made from. *)
let message_of reason =
  match Flambda.Expr.descr (Flambda.Expr.create_invalid reason) with
  | Invalid { message } -> message
  | Let _ | Let_cont _ | Apply _ | Apply_cont _ | Switch _ -> assert false

(* The metadata of the first code that a unit binds at toplevel. *)
let rec metadata_of_some_code expr =
  match Flambda.Expr.descr expr with
  | Let let_expr -> (
    let of_body () =
      Flambda.Let.pattern_match let_expr ~f:(fun _ ~body ->
          metadata_of_some_code body)
    in
    match Flambda.Let.defining_expr let_expr with
    | Static_consts group -> (
      match
        List.find_map
          (fun (const : Flambda.Static_const_or_code.t) ->
            match const with
            | Code code -> Some (Code.code_metadata code)
            | Deleted_code | Static_const _ -> None)
          (Flambda.Static_const_group.to_list group)
      with
      | Some metadata -> metadata
      | None -> of_body ())
    | Simple _ | Prim _ | Set_of_closures _ | Rec_info _ -> of_body ())
  | Let_cont _ | Apply _ | Apply_cont _ | Switch _ | Invalid _ ->
    failwith "no code"

(* A unit that is one invalid expression. *)
let invalid reason =
  unit
    (fun
      ~return_continuation:_ ~exn_continuation:_ ~toplevel_my_alloc_region:_ ->
      Flambda.Expr.create_invalid reason)

(* A unit that is one invalid expression whose message is the given name,
   written by the printer of texts and read back by the reader: the text, and
   what the comparison says of the unit and the unit read. *)
let written_and_read_back name =
  let unit1 = invalid (Message name) in
  let text =
    Flambda2_ui.Flambda_colours.without_colours ~f:(fun () ->
        Format.asprintf "%a" Flambda2_parser.Print_fexpr.flambda_unit
          (Flambda2_parser.Flambda_to_fexpr.conv unit1))
  in
  let file = Filename.temp_file "compare_kinds" ".fl" in
  Out_channel.with_open_bin file (fun channel ->
      Out_channel.output_string channel text);
  let read =
    match Flambda2_parser.Parse_flambda.parse file with
    | Ok unit2 -> answer unit1 unit2
    | Error _ -> "NOT READ BACK"
    | exception exn -> "STOPPED: " ^ Printexc.to_string exn
  in
  Sys.remove file;
  String.trim text, read

(* The words that begin with a capital and are keywords of the text: the names
   of the static constants. *)
let keywords_with_a_capital =
  [ "Block";
    "Value_array";
    "Int_array";
    "Int8_array";
    "Int16_array";
    "Int32_array";
    "Int64_array";
    "Nativeint_array";
    "Float_array";
    "Float32_array";
    "Vec128_array";
    "Vec256_array";
    "Vec512_array";
    "Float_block";
    "Empty_array" ]

let () =
  let continuation1 = Continuation.create () in
  let continuation2 = Continuation.create () in
  let apply =
    Flambda.Apply.create ~callee:(Some argument)
      ~continuation:(Return continuation1)
      (Exn_continuation.create ~exn_handler:continuation2 ~extra_args:[])
      ~args:[argument] ~args_arity:any ~return_arity:any
      ~call_kind:Call_kind.indirect_function_call_unknown_arity
      ~return_mode:
        (Alloc_mode.For_applications.not_alloc_stack
           ~alloc_region:(Variable.create "region" K.region))
      Debuginfo.none ~inlined:Default_inlined
      ~inlining_state:(Inlining_state.default ~round:0)
      ~probe:None ~position:Normal
      ~relative_history:Inlining_history.Relative.empty
  in
  let bound =
    Bound_pattern.singleton
      (Bound_var.create
         (Variable.create "bound" K.value)
         Flambda_debug_uid.none Name_mode.normal)
  in
  let code_metadata =
    match Flambda2_parser.Parse_flambda.parse Sys.argv.(1) with
    | Ok unit -> metadata_of_some_code (Flambda_unit.body unit)
    | Error _ -> failwith ("cannot read " ^ Sys.argv.(1))
  in
  let reasons : (string * Flambda.Invalid.t) list =
    [ "", Body_of_unreachable_continuation continuation1;
      "", Apply_cont_of_unreachable_continuation continuation1;
      "", Defining_expr_of_let (bound, Flambda.Named.create_simple argument);
      "", Closure_type_was_invalid apply;
      ( "",
        Direct_application_parameter_kind_mismatch
          { params_arity = any; args_arity = any; apply } );
      "", Application_argument_kind_mismatch (Flambda_arity.unarize_t any, apply);
      "", Application_result_kind_mismatch (any, apply);
      "", Application_result_kind_mismatch_in_lambda somewhere;
      "", Partial_application_mode_mismatch (apply, code_metadata);
      "", Partial_application_mode_mismatch_in_lambda somewhere;
      "", Calling_local_returning_closure_with_normal_apply apply;
      "", Zero_switch_arms;
      "", Code_not_rebuilt;
      "", To_cmm_dummy_body;
      "", Application_never_returns apply;
      "", Over_application_never_returns apply;
      "Message, of a sentence", Message "Dead variable in apply cont: k (x)";
      "Message, of nothing", Message "";
      ( "Message, of the message of another",
        Message (message_of (Body_of_unreachable_continuation continuation1)) );
      "Message, of a name that no constructor has", Message "(Not_yet 1)" ]
  in
  List.iter
    (fun (what, reason) ->
      let message = message_of reason in
      let name = Flambda2_compare.Compare.name_of_invalid reason in
      let kind = Flambda2_compare.Compare.kind_of_invalid_message message in
      let for_the_text = Flambda2_parser.Invalid_kind.of_message message in
      let show kind = match kind with Some name -> name | None -> "none" in
      Printf.printf "the kind of an invalid expression made from %s: %s%s%s\n"
        (match name with Some name -> name | None -> what)
        (show kind)
        (match name with
        | Some _ when not (Option.equal String.equal name kind) ->
          " (NOT THE NAME OF THE CONSTRUCTOR)"
        | Some _ | None -> "")
        (if Option.equal String.equal kind for_the_text
         then ""
         else " (THE PARSER READS ANOTHER)"))
    reasons;
  Printf.printf
    "two invalid expressions of one kind, about two continuations: %s\n"
    (answer
       (invalid (Body_of_unreachable_continuation continuation1))
       (invalid (Body_of_unreachable_continuation continuation2)));
  Printf.printf "two invalid expressions of two kinds: %s\n"
    (answer
       (invalid (Body_of_unreachable_continuation continuation1))
       (invalid (Apply_cont_of_unreachable_continuation continuation1)));
  Printf.printf "two invalid expressions with two sentences: %s\n"
    (answer
       (invalid (Message "Dead variable in apply cont: k (x)"))
       (invalid (Message "Invalid from fexpr")));
  Printf.printf "an invalid expression with a sentence and one of a kind: %s\n"
    (answer (invalid (Message "Invalid from fexpr")) (invalid Zero_switch_arms));
  Printf.printf
    "an invalid expression of a kind and one made from its message: %s\n"
    (answer
       (invalid (Body_of_unreachable_continuation continuation1))
       (invalid
          (Message (message_of (Body_of_unreachable_continuation continuation2)))));
  Printf.printf
    "two invalid expressions of one name that no constructor has: %s\n"
    (answer (invalid (Message "(Not_yet 1)")) (invalid (Message "Not_yet")));
  Printf.printf
    "two invalid expressions of two names that no constructor has: %s\n"
    (answer
       (invalid (Message "(Not_yet 1)"))
       (invalid (Message "(Nor_this 1)")));
  Printf.printf
    "an invalid expression whose message is the word Message and one with a \
     sentence: %s\n"
    (answer
       (invalid (Message "Message"))
       (invalid (Message "Invalid from fexpr")));
  List.iter
    (fun name ->
      let text, read = written_and_read_back name in
      Printf.printf "the kind %s%s, written and read back: \"%s\", %s\n" name
        (if Flambda2_parser.Flambda_lex.is_keyword name
         then " (a keyword of the text)"
         else "")
        text read)
    (List.filter_map
       (fun (_, reason) -> Flambda2_compare.Compare.name_of_invalid reason)
       reasons
    @ ["Message"; "Not_yet"] @ keywords_with_a_capital)
