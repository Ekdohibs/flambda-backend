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

(* The provers of the compiler, and reification, against their Rocq model.

   A prover is a question that Simplify asks of a type in a typing environment
   (middle_end/flambda2/types/provers.ml), and reification turns a type into
   what a term can be built from (reify.ml). The model has a transcription of
   each prover but six, and of reification without its cases of arrays
   (FlambdaProvers.v, FlambdaReify.v, extracted). Here each prover that the
   model has, and reification, is asked, on generated environments and types, of
   the compiler and of the model, and the two answers are compared, as answers
   of one type (prover_answers.ml, which is to be read with the converter of
   types).

   The environments. The model's provers take an environment of the model, and
   there is no conversion of environments: the two are made by the same calls.
   Each has the same variables and symbols, defined in the same order with the
   same name modes, and then equations added by [add_equation]: the compiler's,
   and the model's transcription of it. A name has at most one equation, or, in
   the runs that say so, at most two, of which the second is met with the first.
   The compiler has two meets, which [add_equation] calls, and the environment
   is made with each; the provers and reification themselves call no meet (the
   provers named [meet_...] answer [Need_meet] where a meet would be needed).
   The model transcribes the n-way meet.

   A prover is asked in the model's environment and in each of the two
   environments of the compiler that is seen the same as the model's: every name
   has the same type, as types of the model, and the same canonical element at
   each of the two name modes, and both are bottom or neither is. Where an
   environment of the compiler is not seen the same, no prover is asked in it:
   the case is left out of the comparison of the provers in that environment and
   counted apart, by two comparisons at the head of each run. What makes the two
   environments differ is not looked for here (types_difftest.ml compares the
   meets).

   An answer. The compiler may stop, with a fatal error or an exception, and the
   model may have no answer ([None]). A stop of the compiler that is caught and
   [None] of the model count as agreement; the reasons are not compared. The
   conversion of the arguments and of the answer runs inside the guard that
   catches a stop ([answer_of_compiler]): that a caught stop comes from the
   function of the compiler is trusted, and the test does not tell. After the
   comparisons of a run, a table says how the compiler answered on the inputs of
   the run, a prover at a time, and why it stopped where it did.

   The kind checks of the compiler ([-flambda2-kind-checks], off by default)
   make a fatal error of a type of a wrong kind, where without them a prover
   answers a default. The model answers the default (FlambdaProvers.v). The runs
   are without them, as the compiler is, but one, which shows the difference as
   a recorded one (C2-2 is the entry of what the compiler does with and without
   them).

   Besides the runs: controls of what the comparisons can see, on the inputs of
   the first run; and one case that no type of the model gives, of two poison
   constants (P1-1).

   What is not compared, the model not having it: [meet_is_flat_float_array],
   [meet_is_non_empty_naked_number_array], [meet_is_immutable_array],
   [prove_is_immutable_array], [prove_is_immediates_array] and the cases of
   arrays of reification (the model has no type of arrays), and [meet_rec_info]
   (the model leaves out the expression of a type of kind [Rec_info]).

   Everything is seeded: the output is the same on every run, and is kept in
   provers_difftest.expected. Case [i] of a run is made from the seed of the run
   plus [i] alone; [provers_difftest.exe -show SEED] shows the case of that seed
   with the two answers of each prover of its run. *)

open Flambda2_identifiers
open Flambda2_kinds
open Flambda2_numbers
open Flambda2_term_basics
open Flambda2_bound_identifiers
open Flambda2_nominal
open Type_comparisons
module K = Flambda_kind
module TG = Flambda2_types__Type_grammar
module TE = Flambda2_types__Typing_env
module ME = Flambda2_types__Meet_env
module P = Flambda2_types__Provers
module Reify = Flambda2_types__Reify
module Binary = Flambda2_types__Meet_and_join
module N_way = Flambda2_types__Meet_and_n_way_join
module M = Flambda2_rocq_semantics.FlambdaModel
module MK = Flambda2_rocq_semantics.FlambdaKindModel
module MT = Flambda2_rocq_semantics.FlambdaTypeGrammar
module MP = Flambda2_rocq_semantics.FlambdaProvers.Provers
module MR = Flambda2_rocq_semantics.FlambdaReify.Reify
module Model_env = Flambda2_rocq_semantics.FlambdaTypingEnv.Typing_env
module Model_meet = Flambda2_rocq_semantics.FlambdaMeet
module Model_join_meet = Flambda2_rocq_semantics.FlambdaJoinMeet
module Model_join_env = Flambda2_rocq_semantics.FlambdaJoinEnv.Join_env
module Model_name_mode = Flambda2_rocq_semantics.FlambdaBindingTime.Name_mode
module Converter = Flambda2_rocq_types_converter.Type_converter
module Model_names = Flambda2_rocq_types_converter.Model_names
module Model_type = Flambda2_rocq_types_converter.Model_type
module N = Flambda2_rocq_types_converter.Model_numbers
module A = Flambda2_rocq_types_converter.Prover_answers
module G = Type_generator
module Random = Minicheck.Splittable_random

(* ======================================================================== *)
(* The names and the two environments                                       *)
(* ======================================================================== *)

type mode =
  | Normal
  | In_types

let mode_of_compiler (mode : mode) : Name_mode.t =
  match mode with Normal -> Name_mode.normal | In_types -> Name_mode.in_types

let mode_of_model (mode : mode) : Model_name_mode.t =
  match mode with
  | Normal -> Model_name_mode.normal
  | In_types -> Model_name_mode.in_types

(* A few names of each sort, made once, numbered in this order; and the
   environment, of the compiler and of the model, that defines the variables and
   the symbols, with nothing known of them. Of the variables of kind value the
   first and the last are defined for types only, and so is the first of the
   naked immediates: an equation makes a variable an alias of one defined after
   it, so a class of aliases may have its earliest member, or a later one, or
   only members, that a term cannot mention. *)
type names =
  { numbering : Model_names.t;
    pool : G.pool;
    (* The variables, in the order of their numbers, which is that of their
       definitions. *)
    variables : (M.variable * MK.flambda_kind * mode) list;
    (* What the two environments define, with the kinds. *)
    defined : (MT.name * MK.flambda_kind) list;
    compiler_env : TE.t;
    model_env : Model_env.t;
    (* An environment that defines the symbols and only some of the variables,
       of the compiler and of the model: for the argument of reification that
       says which variables may be in what is lifted. *)
    fewer_in_compiler : TE.t;
    fewer_in_model : Model_env.t;
    (* The kind of each value slot, which the compiler reads off the slot and
       the model takes as an argument. *)
    value_slot_kinds : (M.value_slot * MK.flambda_kind) list
  }

let create_names () : names =
  let numbering = Model_names.create () in
  let unit = Current_unit.get_cu_exn () in
  let variables : (Variable.t * mode) list =
    List.map
      (fun (name, kind, mode) -> Variable.create name kind, mode)
      [ "w", K.value, In_types;
        "x", K.value, Normal;
        "y", K.value, Normal;
        "z", K.value, Normal;
        "u", K.value, In_types;
        "k", K.naked_immediate, In_types;
        "i", K.naked_immediate, Normal;
        "j", K.naked_immediate, Normal;
        "f", K.naked_float, Normal;
        "r", K.region, Normal;
        "d", K.rec_info, Normal ]
  in
  let symbols =
    List.map
      (fun name -> Symbol.create unit (Linkage_name.of_string name))
      ["s"; "t"]
  in
  let code_ids =
    List.map
      (fun name -> Code_id.create ~name ~debug:Debuginfo.none unit)
      ["c"; "c'"]
  in
  let function_slots =
    List.map (fun name -> Function_slot.create unit ~name ~size:2) ["g"; "h"]
  in
  let value_slots =
    List.map
      (fun (name, kind) ->
        Value_slot.create unit ~name ~is_always_immediate:false kind)
      ["a", K.value; "b", K.value; "n", K.naked_immediate]
  in
  let number (sort : _ Model_names.sort) = List.map sort.number in
  let variables : (M.variable * MK.flambda_kind * mode * Variable.t) list =
    List.map
      (fun (variable, mode) ->
        ( numbering.variables.number variable,
          Converter.kind_to_model (Variable.kind variable),
          mode,
          variable ))
      variables
  in
  let symbols = List.combine (number numbering.symbols symbols) symbols in
  let env_of_compiler variables =
    let env =
      List.fold_left
        (fun env (_, _, mode, variable) ->
          TE.add_definition env
            (Bound_name.create_var
               (Bound_var.create variable Flambda_debug_uid.none
                  (mode_of_compiler mode)))
            (Variable.kind variable))
        (TE.create ~machine_width ~resolver:(fun _ -> None))
        variables
    in
    List.fold_left
      (fun env (_, symbol) ->
        TE.add_definition env (Bound_name.create_symbol symbol) K.value)
      env symbols
  in
  let env_of_model variables =
    let env =
      List.fold_left
        (fun env (x, kind, mode, _) ->
          Model_env.add_variable_definition env x kind (mode_of_model mode))
        Model_env.create variables
    in
    List.fold_left
      (fun env (symbol, _) -> Model_env.add_symbol_definition env symbol)
      env symbols
  in
  let compiler_env = env_of_compiler variables in
  let model_env = env_of_model variables in
  (* Every other variable, from the first. *)
  let fewer = List.filteri (fun i _ -> i mod 2 = 0) variables in
  let of_kind (kind : MK.flambda_kind) =
    List.filter_map
      (fun (x, kind', _, _) -> if Stdlib.( = ) kind kind' then Some x else None)
      variables
  in
  let value_slot_kinds =
    List.map
      (fun slot ->
        ( numbering.value_slots.number slot,
          Converter.kind_to_model (Value_slot.kind slot) ))
      value_slots
  in
  { numbering;
    pool =
      { value_variables = of_kind Value;
        naked_immediate_variables = of_kind (Naked_number Kind_naked_immediate);
        naked_float_variables = of_kind (Naked_number Kind_naked_float);
        region_variables = of_kind Region;
        rec_info_variables = of_kind Rec_info;
        symbols = List.map fst symbols;
        code_ids = number numbering.code_ids code_ids;
        function_slots = number numbering.function_slots function_slots;
        value_slots = List.map fst value_slot_kinds;
        unknown_name = N.positive_of_int 1000
      };
    variables = List.map (fun (x, kind, mode, _) -> x, kind, mode) variables;
    defined =
      List.map
        (fun (x, kind, _, _) : (MT.name * _) -> Name_var x, kind)
        variables
      @ List.map
          (fun (symbol, _) : (MT.name * _) -> Name_symbol symbol, MK.Value)
          symbols;
    compiler_env;
    model_env;
    fewer_in_compiler = env_of_compiler fewer;
    fewer_in_model = env_of_model fewer;
    value_slot_kinds
  }

(* The two meets of the compiler, as [add_equation] takes them, each with the
   setting of the flag that chooses it. *)
type implementation =
  { implementation : string;
    meet_expanded_head : ME.meet_expanded_head;
    algorithm : Oxcaml_flags.join_algorithm
  }

let n_way : implementation =
  { implementation = "n-way";
    meet_expanded_head = N_way.meet_expanded_head;
    algorithm = N_way
  }

let binary : implementation =
  { implementation = "binary";
    meet_expanded_head = Binary.meet_expanded_head;
    algorithm = Binary
  }

(* [add_equation] of the model: the transcription of the compiler's, with the
   choices that are the compiler's, the depth of joins that the compiler reads
   from its flag, and fuel. *)
let add_equation_of_model :
    Model_env.t -> MT.name -> MT.flambda_type -> Model_env.t option =
  let fuel = N.nat_of_int 200 in
  let fuels : Model_join_meet.fuels =
    { fuel_of_env = fuel; fuel_of_types = fuel; fuel_of_meet = fuel }
  in
  fun env name t ->
    Model_join_meet.model_add_equation Model_join_env.compiler_choices
      Model_meet.compiler_choices
      (N.nat_of_int (Flambda2_ui.Flambda_features.join_depth ()))
      fuels env name t

let to_compiler (names : names) (t : MT.flambda_type) : TG.t =
  Converter.type_of_model names.numbering machine_width t

(* The environment of the model after the equations. *)
let model_env_after (names : names)
    (equations : (MT.name * MT.flambda_type) list) :
    (Model_env.t, string) result =
  let rec add env equations : (Model_env.t, string) result =
    match equations with
    | [] -> Ok env
    | (name, t) :: equations -> (
      match add_equation_of_model env name t with
      | Some env -> add env equations
      | None -> Error "the model has no answer where an equation is added")
  in
  add names.model_env equations

(* The environment of the compiler after the equations, with one of its two
   meets. *)
let compiler_env_after (names : names) (implementation : implementation)
    (equations : (MT.name * MT.flambda_type) list) : (TE.t, string) result =
  let meet_expanded_head = implementation.meet_expanded_head in
  let before = !Oxcaml_flags.Flambda2.join_algorithm in
  Oxcaml_flags.Flambda2.join_algorithm
    := Oxcaml_flags.Set implementation.algorithm;
  Fun.protect
    ~finally:(fun () -> Oxcaml_flags.Flambda2.join_algorithm := before)
    (fun () ->
      match
        Misc.Colours.without_colours ~f:(fun () ->
            or_fatal_error (fun () ->
                List.fold_left
                  (fun env (name, t) ->
                    ME.use_meet_env ~meet_expanded_head env ~f:(fun env ->
                        ME.add_equation env
                          (Converter.name_of_model names.numbering name)
                          (to_compiler names t) ~meet_expanded_head))
                  names.compiler_env equations))
      with
      | Ok env -> Ok env
      | Error line ->
        Error ("a fatal error where an equation is added: " ^ general line))

(* What differs in what is seen of the two environments, if anything does: for
   each name, its type, as a type of the model, and its canonical element at
   each of the two name modes; and whether the environment is bottom. *)
let seen_differently (names : names) (model_env : Model_env.t) (env : TE.t) :
    string option =
  let numbering = names.numbering in
  let of_name ((name : MT.name), kind) : string option =
    let name_of_compiler = Converter.name_of_model numbering name in
    let type_differs =
      match Model_env.find model_env name None with
      | None -> true
      | Some of_model ->
        not
          (Model_type.same of_model
             (Converter.type_to_model numbering
                (TE.find env name_of_compiler
                   (Some (Converter.kind_of_model kind)))))
    in
    let canonical_differs mode =
      let of_model =
        Model_env.get_canonical_simple_exn model_env
          (Some (mode_of_model mode))
          None (MT.simple_of_name name)
      in
      let of_compiler =
        match
          TE.get_canonical_simple_exn env ~min_name_mode:(mode_of_compiler mode)
            (Simple.name name_of_compiler)
        with
        | canonical -> Some (Converter.simple_to_model numbering canonical)
        | exception Not_found -> None
      in
      match of_model with
      | None -> true
      | Some of_model -> not (Stdlib.( = ) of_model of_compiler)
    in
    if type_differs
    then Some "the type of a name"
    else if canonical_differs Normal || canonical_differs In_types
    then Some "the canonical element of a name"
    else None
  in
  if not (Bool.equal (Model_env.is_bottom model_env) (TE.is_bottom env))
  then Some "one is bottom"
  else List.find_map of_name names.defined

(* ======================================================================== *)
(* The inputs                                                               *)
(* ======================================================================== *)

(* The environment that says which variables may be in what reification lifts:
   the environment in which the type is reified, or one that defines fewer
   variables. *)
type allowed_in =
  | The_environment
  | Fewer_variables

(* The arguments of the provers that take more than a type, and of
   reification. *)
type arguments =
  { min_name_mode : mode;
    kind : MK.flambda_kind;
    field_index : int;
    function_slot : M.function_slot;
    value_slot : M.value_slot;
    variable : M.variable;
    (* Of reification: the variables that are defined at toplevel, and those
       that are projections from symbols. *)
    allowed_in : allowed_in;
    at_toplevel : M.variable list;
    symbol_projections : M.variable list
  }

(* An environment of the compiler, and whether a prover can be asked in it. *)
type compiler_env =
  | Seen_the_same of TE.t
  | Not_asked of string

type environments =
  { model : (Model_env.t, string) result;
    with_n_way : compiler_env;
    with_binary : compiler_env
  }

type input =
  { equations : (MT.name * MT.flambda_type) list;
    asked : MT.flambda_type;
    (* The second type, of the one prover that takes two. *)
    other : MT.flambda_type;
    arguments : arguments;
    (* Made from the fields above, once. *)
    asked_of_compiler : TG.t Lazy.t;
    other_of_compiler : TG.t Lazy.t;
    environments : environments Lazy.t
  }

let input (names : names) ~equations ~asked ~other ~arguments : input =
  let environments =
    lazy
      (let model = model_env_after names equations in
       let compiler implementation : compiler_env =
         match model, compiler_env_after names implementation equations with
         | Error why, Error (_ : string) ->
           Not_asked ("neither makes the environment: " ^ why)
         | Error why, Ok (_ : TE.t) -> Not_asked why
         | Ok (_ : Model_env.t), Error why -> Not_asked why
         | Ok model_env, Ok env -> (
           match seen_differently names model_env env with
           | None -> Seen_the_same env
           | Some what ->
             Not_asked ("the two environments are not seen the same: " ^ what))
       in
       { model; with_n_way = compiler n_way; with_binary = compiler binary })
  in
  { equations;
    asked;
    other;
    arguments;
    asked_of_compiler = lazy (to_compiler names asked);
    other_of_compiler = lazy (to_compiler names other);
    environments
  }

(* What every comparison needs of an input: its types, as types of the compiler.
   An input whose types the conversion cannot construct as types of the compiler
   is refused here. *)
let prepare (input : input) : input =
  ignore (Lazy.force input.asked_of_compiler : TG.t);
  ignore (Lazy.force input.other_of_compiler : TG.t);
  input

let print_mode ppf (mode : mode) =
  Format.pp_print_string ppf
    (match mode with Normal -> "normal" | In_types -> "in types")

(* What is shown of an input: its second type, and which of its arguments. *)
type shown =
  { two_types : bool;
    of_reification : bool
  }

let print_input (shown : shown) ppf
    { equations;
      asked;
      other;
      arguments =
        { min_name_mode;
          kind;
          field_index;
          function_slot;
          value_slot;
          variable;
          allowed_in;
          at_toplevel;
          symbol_projections
        };
      asked_of_compiler = _;
      other_of_compiler = _;
      environments = _
    } =
  Format.fprintf ppf "@[<v>%a%a%a@ %t@]"
    (fun ppf ->
      List.iter (fun (name, t) ->
          Format.fprintf ppf "with %a : %a@ " Model_type.print_name name
            Model_type.print t))
    equations Model_type.print asked
    (fun ppf other ->
      if shown.two_types
      then Format.fprintf ppf "@ and@ %a" Model_type.print other)
    other
    (fun ppf ->
      let variables = Model_type.print_list Model_type.print_variable in
      if shown.of_reification
      then
        Format.fprintf ppf
          "@[<hov 2>with the arguments:@ the variables allowed are those of \
           %s,@ at toplevel %a,@ projections from symbols %a@]"
          (match allowed_in with
          | The_environment -> "the environment"
          | Fewer_variables -> "an environment with fewer")
          variables at_toplevel variables symbol_projections
      else
        Format.fprintf ppf
          "@[<hov 2>with the arguments:@ min_name_mode %a,@ kind %a,@ field \
           %d,@ function slot %a,@ value slot %a,@ variable %a@]"
          print_mode min_name_mode Model_type.print_kind kind field_index
          N.print_positive function_slot N.print_positive value_slot
          Model_type.print_variable variable)

let shrink_input (names : names)
    ({ equations; asked; other; arguments; _ } as _input : input) : input list =
  let input = input names ~arguments in
  List.mapi
    (fun i _ ->
      input
        ~equations:(List.filteri (fun j _ -> i <> j) equations)
        ~asked ~other)
    equations
  @ List.map (fun asked -> input ~equations ~asked ~other) (G.shrink asked)
  @ List.map (fun other -> input ~equations ~asked ~other) (G.shrink other)

(* ---- The generator ----------------------------------------------------- *)

let config : G.config = { well_formed = true; depth = 2 }

let naked_immediate : MK.flambda_kind = Naked_number Kind_naked_immediate

(* An alias to a [Simple] of a kind: a variable, a symbol or a constant. *)
let alias (names : names) r (kind : MK.flambda_kind) : MT.flambda_type =
  match G.simple config r names.pool kind with
  | Some simple -> MT.alias_type_of kind simple
  | None -> MT.unknown kind

let small_numbers r ~max =
  G.small_number r :: G.list r ~max (fun () -> G.small_number r)

(* A type of naked immediates: numbers, the result of a relation on a name, an
   alias, or what the generator of types makes. *)
let naked_immediates (names : names) r : MT.flambda_type =
  let name () = G.name config r names.pool in
  match G.below r 7 with
  | 0 -> MT.these_naked_immediates (small_numbers r ~max:2)
  | 1 -> MT.is_int_for_scrutinee (name ())
  | 2 -> MT.get_tag_for_block (name ())
  | 3 -> MT.is_null_for_scrutinee (name ())
  | 4 -> alias names r naked_immediate
  | _ -> G.type_of_kind config r names.pool naked_immediate

(* A type of kind value that a creation function of the model makes, of the
   shapes that the provers ask for: a tagged immediate, a boxed number, a block,
   a closure, a string, with aliases where a prover answers a [Simple]. *)
let value_of_a_shape (names : names) r : MT.flambda_type =
  let pool = names.pool in
  let mode () = G.alloc_mode config r in
  let scannable_tag () = N.z_of_int (G.pick r [0; 1; 2; 250]) in
  match G.below r 15 with
  | 0 -> MT.tagged_immediate_alias_to (G.pick r pool.naked_immediate_variables)
  | 1 -> MT.tag_immediate (naked_immediates names r)
  | 2 -> MT.these_tagged_immediates (small_numbers r ~max:2)
  | 3 | 4 ->
    let number = G.pick r G.all_boxable_numbers in
    MT.box_number number (alias names r (MK.unboxed_kind number)) (mode ())
  | 5 ->
    MT.these_boxed_numbers
      (G.pick r G.all_boxable_numbers)
      (small_numbers r ~max:1) (mode ())
  | 6 | 7 ->
    (* A block of one tag and one size, of which every field is an alias. *)
    let shape = G.block_shape r in
    let size =
      match shape with
      | Scannable (Mixed_record (value_prefix_size, flat_suffix)) ->
        N.int_of_nat value_prefix_size + List.length flat_suffix
      | Scannable Value_only | Float_record -> G.below r 4
    in
    let tag =
      match shape with
      | Scannable (Value_only | Mixed_record _) -> scannable_tag ()
      | Float_record -> MT.double_array_tag
    in
    MT.immutable_block (G.chance r ~percent:10) tag shape (mode ())
      (List.map (alias names r) (MT.field_kinds shape (N.nat_of_int size)))
  | 8 -> (
    match
      MT.immutable_block_with_size_at_least
        (if G.chance r ~percent:50 then Known (scannable_tag ()) else Unknown)
        (N.nat_of_int (1 + G.below r 3))
        (Scannable Value_only)
        (G.pick r pool.value_variables)
    with
    | Some t -> t
    | None -> MT.any_block)
  | 9 | 10 ->
    (* A closure at a function slot of a set, with its function type, the
       closure at that slot and a value slot, each an alias. *)
    let slot = G.pick r pool.function_slots in
    let function_types :
        (M.function_slot * MT.function_type MT.Or_unknown.t) list =
      match G.below r 5 with
      | 0 -> []
      | 1 -> [slot, Unknown]
      | _ -> [slot, Known { function_type_code_id = G.pick r pool.code_ids }]
    in
    let value_slot = G.pick r pool.value_slots in
    (if G.chance r ~percent:50
     then MT.exactly_this_closure
     else MT.at_least_this_closure)
      slot function_types
      [G.pick r pool.function_slots, alias names r Value]
      [ ( value_slot,
          alias names r
            (if G.chance r ~percent:85
             then List.assoc value_slot names.value_slot_kinds
             else MK.Value) ) ]
      (mode ())
  | 11 ->
    (* The type of the value slot may be of another kind than the slot. *)
    let kind, variables =
      if G.chance r ~percent:80
      then MK.Value, pool.value_variables
      else naked_immediate, pool.naked_immediate_variables
    in
    MT.closure_with_at_least_this_value_slot
      (G.pick r pool.function_slots)
      (G.pick r pool.value_slots)
      (G.pick r variables) kind
  | 12 ->
    MT.this_immutable_string
      (G.list r ~max:3 (fun () -> N.z_of_int (97 + G.below r 3)))
  | 13 -> (
    match G.below r 5 with
    | 0 -> MT.mutable_block (mode ())
    | 1 -> MT.any_block
    | 2 ->
      MT.blocks_with_these_tags [scannable_tag (); scannable_tag ()] (mode ())
    | 3 -> MT.null
    | _ -> MT.any_tagged_immediate)
  | _ -> alias names r Value

(* A type of a kind: half the time of one of the shapes above, where there are
   some, and otherwise what the generator of types makes. *)
let type_of_kind (names : names) r (kind : MK.flambda_kind) : MT.flambda_type =
  let general () = G.type_of_kind config r names.pool kind in
  match kind with
  | Value ->
    if G.chance r ~percent:50 then value_of_a_shape names r else general ()
  | Naked_number number -> (
    if G.is_naked_immediate number
    then naked_immediates names r
    else
      match G.below r 4 with
      | 0 -> MT.these_naked_numbers number (small_numbers r ~max:2)
      | 1 -> alias names r kind
      | _ -> general ())
  | Region | Rec_info -> general ()

(* Equations on names of the tests, a name having at most one. The type of a
   variable mentions no variable but those numbered after it, and the type of a
   symbol no variable: no type is reached from itself. The variables that are
   regions or of kind [Rec_info] have none: the conversion makes no head of kind
   [Rec_info] of the compiler from the model's. *)
let rec generate_equations ?(twice = false) (names : names) r :
    (MT.name * MT.flambda_type) list =
  if twice
  then
    (* A name may have two: the second is met with the first. *)
    let first = generate_equations names r in
    first @ generate_equations names r
  else generate_equations_once names r

and generate_equations_once (names : names) r : (MT.name * MT.flambda_type) list
    =
  let equation (name : MT.name) kind ~(allowed : M.variable -> bool) =
    let rec attempt n =
      if n = 0
      then None
      else
        let t = type_of_kind names r kind in
        if List.for_all allowed (MT.free_variables t)
        then Some (name, t)
        else attempt (n - 1)
    in
    if G.chance r ~percent:55 then attempt 6 else None
  in
  List.filter_map
    (fun (x, (kind : MK.flambda_kind), (_ : mode)) ->
      match kind with
      | Value | Naked_number _ ->
        equation (Name_var x) kind ~allowed:(fun y ->
            N.compare_positive x y < 0)
      | Region | Rec_info -> None)
    names.variables
  @ List.filter_map
      (fun symbol ->
        if G.chance r ~percent:50
        then equation (Name_symbol symbol) Value ~allowed:(fun _ -> false)
        else None)
      names.pool.symbols

let generate_arguments (names : names) r ~(asked : MT.flambda_type) : arguments
    =
  let pool = names.pool in
  let some_variables ~percent =
    List.filter_map
      (fun (x, (_ : MK.flambda_kind), (_ : mode)) ->
        if G.chance r ~percent then Some x else None)
      names.variables
  in
  { min_name_mode = (if G.chance r ~percent:60 then Normal else In_types);
    (* For the prover that takes the kind of the type, the kind of the type,
       most of the time; for the one that takes the kind of a field, the kind of
       values, most of the time. *)
    kind =
      (match G.below r 10 with
      | 0 -> G.kind r
      | 1 | 2 | 3 -> MK.Value
      | _ -> MT.kind asked);
    field_index = G.below r 4;
    function_slot = G.pick r pool.function_slots;
    value_slot = G.pick r pool.value_slots;
    variable =
      (let x, (_ : MK.flambda_kind), (_ : mode) = G.pick r names.variables in
       x);
    allowed_in =
      (if G.chance r ~percent:70 then The_environment else Fewer_variables);
    at_toplevel = some_variables ~percent:65;
    symbol_projections = some_variables ~percent:20
  }

(* An alias to a name of the tests, of the kind given if there is one. *)
let alias_to_a_name (names : names) r (kind : MK.flambda_kind) : MT.flambda_type
    =
  let symbol =
    match kind with
    | Value -> G.chance r ~percent:25
    | Naked_number _ | Region | Rec_info -> false
  in
  if symbol
  then MT.alias_type_of kind (Symbol (G.pick r names.pool.symbols))
  else
    match G.variables_of_kind names.pool kind with
    | [] -> MT.unknown kind
    | _ :: _ as variables -> MT.alias_type_of kind (Var (G.pick r variables))

(* A type to ask a prover of: an alias to a name of the tests, so that the
   prover reads the environment, or a type as above. *)
let asked_of_kind (names : names) r (kind : MK.flambda_kind) : MT.flambda_type =
  if G.chance r ~percent:40
  then alias_to_a_name names r kind
  else type_of_kind names r kind

(* An input whose type is usually of kind value. With [twice], a name may have
   two equations. *)
let generate_on_values ?twice (names : names) r : input =
  let equations = generate_equations ?twice names r in
  let kind = if G.chance r ~percent:92 then MK.Value else G.kind r in
  let asked = asked_of_kind names r kind in
  let arguments = generate_arguments names r ~asked in
  input names ~equations ~asked ~other:asked ~arguments

(* An input whose type is of naked numbers: of each of the twelve kinds as
   often, and a few times of a kind that [G.kind] chooses. *)
let generate_on_naked_numbers (names : names) r : input =
  let equations = generate_equations names r in
  let kind : MK.flambda_kind =
    if G.chance r ~percent:94
    then Naked_number (G.pick r G.all_naked_number_kinds)
    else G.kind r
  in
  let asked = asked_of_kind names r kind in
  let arguments = generate_arguments names r ~asked in
  input names ~equations ~asked ~other:asked ~arguments

(* An input whose type is of naked immediates with relations to names, up to
   four, or the tagging of one: the provers of immediates ask the provers of
   [Is_int], [Get_tag] and [Is_null] of each name, in the environment. *)
let generate_on_relations (names : names) r : input =
  let equations = generate_equations names r in
  let immediates : MT.flambda_type =
    Type_naked_immediate
      (Ok
         (No_alias
            { naked_immediates =
                (if G.chance r ~percent:50
                 then Unknown
                 else Known (small_numbers r ~max:3));
              inverse_relations =
                (G.relation r, G.name config r names.pool)
                :: G.list r ~max:3 (fun () ->
                    G.relation r, G.name config r names.pool)
            }))
  in
  let asked =
    if G.chance r ~percent:50 then immediates else MT.tag_immediate immediates
  in
  let arguments = generate_arguments names r ~asked in
  input names ~equations ~asked ~other:asked ~arguments

(* An input whose type is raw: of any kind, as the generator of types makes it
   without keeping what the creation functions of the compiler keep (a product
   wider than its index, a boxed number that holds a type of another kind, a set
   of closures with no function type for a closure). The equations are as above.
   Many are inputs whose types the conversion cannot construct, and are
   refused. *)
let generate_raw (names : names) r : input =
  let equations = generate_equations names r in
  let asked = G.flambda_type { well_formed = false; depth = 3 } r names.pool in
  let other =
    if G.chance r ~percent:50
    then asked
    else G.flambda_type { well_formed = false; depth = 2 } r names.pool
  in
  let arguments = generate_arguments names r ~asked in
  input names ~equations ~asked ~other ~arguments

(* An input whose type is of any kind, as [G.kind] chooses it: of kind value
   more often than not. *)
let generate_on_any_kind (names : names) r : input =
  let equations = generate_equations names r in
  let asked = asked_of_kind names r (G.kind r) in
  let other =
    if G.chance r ~percent:50 then asked else asked_of_kind names r (G.kind r)
  in
  let arguments = generate_arguments names r ~asked in
  input names ~equations ~asked ~other ~arguments

(* An input with two types, of kind value most of the time: two types as above,
   or one type twice. *)
let generate_two_types (names : names) r : input =
  let equations = generate_equations names r in
  let kind () = if G.chance r ~percent:95 then MK.Value else G.kind r in
  let asked = asked_of_kind names r (kind ()) in
  let other =
    if G.chance r ~percent:15 then asked else asked_of_kind names r (kind ())
  in
  let arguments = generate_arguments names r ~asked in
  input names ~equations ~asked ~other ~arguments

(* ======================================================================== *)
(* The provers                                                              *)
(* ======================================================================== *)

(* A prover: its name in provers.mli, and its answer on an input, of the
   compiler in an environment of the compiler and of the model in the
   environment of the model. *)
type prover =
  { prover : string;
    of_compiler : TE.t -> input -> A.answer;
    of_model : Model_env.t -> input -> A.answer
  }

let provers (names : names) : prover list =
  let numbering = names.numbering in
  let asked input = Lazy.force input.asked_of_compiler in
  let module C = A.Of_compiler in
  let module O = A.Of_model in
  (* A prover of an environment and a type. *)
  let proof prover of_compiler value_of_compiler of_model value_of_model =
    { prover;
      of_compiler =
        (fun env input ->
          C.proof value_of_compiler (of_compiler env (asked input)));
      of_model =
        (fun env input -> O.proof value_of_model (of_model env input.asked))
    }
  in
  let shortcut prover of_compiler value_of_compiler of_model value_of_model =
    { prover;
      of_compiler =
        (fun env input ->
          C.shortcut value_of_compiler (of_compiler env (asked input)));
      of_model =
        (fun env input -> O.shortcut value_of_model (of_model env input.asked))
    }
  in
  (* A prover that takes the least name mode of the [Simple] it answers. *)
  let shortcut_with_mode prover of_compiler of_model =
    { prover;
      of_compiler =
        (fun env input ->
          C.shortcut (C.simple numbering)
            (of_compiler env
               ~min_name_mode:(mode_of_compiler input.arguments.min_name_mode)
               (asked input)));
      of_model =
        (fun env input ->
          O.shortcut O.simple
            (of_model env
               (mode_of_model input.arguments.min_name_mode)
               input.asked))
    }
  in
  let simple = C.simple numbering in
  let variable_of_model = Converter.variable_of_model numbering in
  let kind_of_value_slot slot = List.assoc slot names.value_slot_kinds in
  [ { prover = "prove_equals_to_simple_of_kind";
      of_compiler =
        (fun env input ->
          C.proof simple
            (P.prove_equals_to_simple_of_kind env (asked input)
               (Converter.kind_of_model input.arguments.kind)));
      of_model =
        (fun env input ->
          O.proof O.simple
            (MP.prove_equals_to_simple_of_kind env input.asked
               input.arguments.kind))
    };
    proof "prove_equals_tagged_immediates" P.prove_equals_tagged_immediates
      C.target_ints MP.prove_equals_tagged_immediates O.numbers;
    shortcut "meet_equals_tagged_immediates" P.meet_equals_tagged_immediates
      C.target_ints MP.meet_equals_tagged_immediates O.numbers;
    shortcut "meet_naked_immediates" P.meet_naked_immediates C.target_ints
      MP.meet_naked_immediates O.numbers;
    shortcut "meet_equals_single_tagged_immediate"
      P.meet_equals_single_tagged_immediate C.number
      MP.meet_equals_single_tagged_immediate O.number;
    shortcut "meet_naked_float32s" P.meet_naked_float32s C.float32s
      MP.meet_naked_float32s O.numbers;
    shortcut "meet_naked_floats" P.meet_naked_floats C.floats
      MP.meet_naked_floats O.numbers;
    shortcut "meet_naked_int8s" P.meet_naked_int8s C.int8s MP.meet_naked_int8s
      O.numbers;
    shortcut "meet_naked_int16s" P.meet_naked_int16s C.int16s
      MP.meet_naked_int16s O.numbers;
    shortcut "meet_naked_int32s" P.meet_naked_int32s C.int32s
      MP.meet_naked_int32s O.numbers;
    shortcut "meet_naked_int64s" P.meet_naked_int64s C.int64s
      MP.meet_naked_int64s O.numbers;
    shortcut "meet_naked_nativeints" P.meet_naked_nativeints C.nativeints
      MP.meet_naked_nativeints O.numbers;
    shortcut "meet_naked_vec128s" P.meet_naked_vec128s C.vec128s
      MP.meet_naked_vec128s O.numbers;
    shortcut "meet_naked_vec256s" P.meet_naked_vec256s C.vec256s
      MP.meet_naked_vec256s O.numbers;
    shortcut "meet_naked_vec512s" P.meet_naked_vec512s C.vec512s
      MP.meet_naked_vec512s O.numbers;
    shortcut "meet_naked_masks" P.meet_naked_masks C.masks MP.meet_naked_masks
      O.numbers;
    shortcut "meet_variant_like" P.meet_variant_like C.variant_like
      MP.meet_variant_like O.variant_like;
    proof "prove_variant_like" P.prove_variant_like C.variant_like
      MP.prove_variant_like O.variant_like;
    proof "prove_is_a_boxed_or_tagged_number"
      P.prove_is_a_boxed_or_tagged_number
      (C.boxed_or_tagged_number numbering)
      MP.prove_is_a_boxed_or_tagged_number O.boxed_or_tagged_number;
    proof "prove_nothing" P.prove_nothing C.unit MP.prove_nothing O.unit;
    proof "prove_is_a_tagged_immediate" P.prove_is_a_tagged_immediate C.unit
      MP.prove_is_a_tagged_immediate O.unit;
    proof "prove_is_a_boxed_float32" P.prove_is_a_boxed_float32 C.unit
      MP.prove_is_a_boxed_float32 O.unit;
    proof "prove_is_a_boxed_float" P.prove_is_a_boxed_float C.unit
      MP.prove_is_a_boxed_float O.unit;
    proof "prove_is_a_boxed_int32" P.prove_is_a_boxed_int32 C.unit
      MP.prove_is_a_boxed_int32 O.unit;
    proof "prove_is_a_boxed_int64" P.prove_is_a_boxed_int64 C.unit
      MP.prove_is_a_boxed_int64 O.unit;
    proof "prove_is_a_boxed_nativeint" P.prove_is_a_boxed_nativeint C.unit
      MP.prove_is_a_boxed_nativeint O.unit;
    proof "prove_is_a_boxed_vec128" P.prove_is_a_boxed_vec128 C.unit
      MP.prove_is_a_boxed_vec128 O.unit;
    proof "prove_is_a_boxed_vec256" P.prove_is_a_boxed_vec256 C.unit
      MP.prove_is_a_boxed_vec256 O.unit;
    proof "prove_is_a_boxed_vec512" P.prove_is_a_boxed_vec512 C.unit
      MP.prove_is_a_boxed_vec512 O.unit;
    proof "prove_is_a_boxed_mask" P.prove_is_a_boxed_mask C.unit
      MP.prove_is_a_boxed_mask O.unit;
    proof "prove_is_or_is_not_a_boxed_float" P.prove_is_or_is_not_a_boxed_float
      C.bool MP.prove_is_or_is_not_a_boxed_float O.bool;
    proof "prove_unique_tag_and_size" P.prove_unique_tag_and_size C.tag_and_size
      MP.prove_unique_tag_and_size O.tag_and_size;
    proof "prove_is_int" P.prove_is_int C.bool MP.prove_is_int O.bool;
    proof "prove_is_not_a_pointer" P.prove_is_not_a_pointer C.bool
      MP.prove_is_not_a_pointer O.bool;
    shortcut "meet_is_int_variant_only" P.meet_is_int_variant_only C.bool
      MP.meet_is_int_variant_only O.bool;
    proof "prove_get_tag" P.prove_get_tag C.tags MP.prove_get_tag O.numbers;
    shortcut "meet_is_null" P.meet_is_null C.bool MP.meet_is_null O.bool;
    proof "prove_unique_fully_constructed_immutable_heap_block"
      P.prove_unique_fully_constructed_immutable_heap_block (C.block numbering)
      MP.prove_unique_fully_constructed_immutable_heap_block O.block;
    shortcut "meet_single_closures_entry" P.meet_single_closures_entry
      (C.closures_entry numbering)
      MP.meet_single_closures_entry O.closures_entry;
    proof "prove_single_closures_entry" P.prove_single_closures_entry
      (C.closures_entry numbering)
      MP.prove_single_closures_entry O.closures_entry;
    shortcut "meet_code_ids" P.meet_code_ids (C.code_ids numbering)
      MP.meet_code_ids O.code_ids;
    proof "prove_code_ids" P.prove_code_ids (C.code_ids numbering)
      MP.prove_code_ids O.code_ids;
    shortcut "meet_strings" P.meet_strings C.strings MP.meet_strings O.strings;
    proof "prove_strings" P.prove_strings C.strings MP.prove_strings O.strings;
    { prover = "prove_tagging_of_simple";
      of_compiler =
        (fun env input ->
          C.proof simple
            (P.prove_tagging_of_simple env
               ~min_name_mode:(mode_of_compiler input.arguments.min_name_mode)
               (asked input)));
      of_model =
        (fun env input ->
          O.proof O.simple
            (MP.prove_tagging_of_simple env
               (mode_of_model input.arguments.min_name_mode)
               input.asked))
    };
    shortcut_with_mode "meet_tagging_of_simple" P.meet_tagging_of_simple
      MP.meet_tagging_of_simple ]
  @ List.map
      (fun (prover, of_compiler, of_model) ->
        shortcut_with_mode prover of_compiler of_model)
      [ ( "meet_boxed_float32_containing_simple",
          P.meet_boxed_float32_containing_simple,
          MP.meet_boxed_float32_containing_simple );
        ( "meet_boxed_float_containing_simple",
          P.meet_boxed_float_containing_simple,
          MP.meet_boxed_float_containing_simple );
        ( "meet_boxed_int32_containing_simple",
          P.meet_boxed_int32_containing_simple,
          MP.meet_boxed_int32_containing_simple );
        ( "meet_boxed_int64_containing_simple",
          P.meet_boxed_int64_containing_simple,
          MP.meet_boxed_int64_containing_simple );
        ( "meet_boxed_nativeint_containing_simple",
          P.meet_boxed_nativeint_containing_simple,
          MP.meet_boxed_nativeint_containing_simple );
        ( "meet_boxed_vec128_containing_simple",
          P.meet_boxed_vec128_containing_simple,
          MP.meet_boxed_vec128_containing_simple );
        ( "meet_boxed_vec256_containing_simple",
          P.meet_boxed_vec256_containing_simple,
          MP.meet_boxed_vec256_containing_simple );
        ( "meet_boxed_vec512_containing_simple",
          P.meet_boxed_vec512_containing_simple,
          MP.meet_boxed_vec512_containing_simple );
        ( "meet_boxed_mask_containing_simple",
          P.meet_boxed_mask_containing_simple,
          MP.meet_boxed_mask_containing_simple ) ]
  @ [ { prover = "meet_block_field_simple";
        of_compiler =
          (fun env input ->
            C.shortcut simple
              (P.meet_block_field_simple env
                 ~min_name_mode:(mode_of_compiler input.arguments.min_name_mode)
                 ~field_kind:(Converter.kind_of_model input.arguments.kind)
                 (asked input)
                 (Target_ocaml_int.of_int machine_width
                    input.arguments.field_index)));
        of_model =
          (fun env input ->
            O.shortcut O.simple
              (MP.meet_block_field_simple env
                 (mode_of_model input.arguments.min_name_mode)
                 input.arguments.kind input.asked
                 (N.nat_of_int input.arguments.field_index)))
      };
      { prover = "meet_project_value_slot_simple";
        of_compiler =
          (fun env input ->
            C.shortcut simple
              (P.meet_project_value_slot_simple env
                 ~min_name_mode:(mode_of_compiler input.arguments.min_name_mode)
                 (asked input)
                 (Converter.value_slot_of_model numbering
                    input.arguments.value_slot)));
        of_model =
          (fun env input ->
            (* The kind that the compiler reads off the value slot. *)
            O.shortcut O.simple
              (MP.meet_project_value_slot_simple env
                 (mode_of_model input.arguments.min_name_mode)
                 input.asked input.arguments.value_slot
                 (kind_of_value_slot input.arguments.value_slot)))
      };
      { prover = "meet_project_function_slot_simple";
        of_compiler =
          (fun env input ->
            C.shortcut simple
              (P.meet_project_function_slot_simple env
                 ~min_name_mode:(mode_of_compiler input.arguments.min_name_mode)
                 (asked input)
                 (Converter.function_slot_of_model numbering
                    input.arguments.function_slot)));
        of_model =
          (fun env input ->
            O.shortcut O.simple
              (MP.meet_project_function_slot_simple env
                 (mode_of_model input.arguments.min_name_mode)
                 input.asked input.arguments.function_slot))
      };
      proof "prove_alloc_mode_of_boxed_number"
        P.prove_alloc_mode_of_boxed_number C.alloc_mode
        MP.prove_alloc_mode_of_boxed_number O.alloc_mode;
      { prover = "never_holds_locally_allocated_values";
        of_compiler =
          (fun env input ->
            C.proof C.unit
              (P.never_holds_locally_allocated_values env
                 (variable_of_model input.arguments.variable)));
        of_model =
          (fun env input ->
            O.proof O.unit
              (MP.never_holds_locally_allocated_values env
                 input.arguments.variable))
      };
      { prover = "prove_physical_equality";
        of_compiler =
          (fun env input ->
            C.proof C.bool
              (P.prove_physical_equality env (asked input)
                 (Lazy.force input.other_of_compiler)));
        of_model =
          (fun env input ->
            O.proof O.bool
              (MP.prove_physical_equality env input.asked input.other))
      } ]

(* Reification ([Reify.reify]), asked as a prover is. Its first three arguments
   are of the input: the environment that says which variables are allowed, and
   the two sets of variables, as functions. *)
let reification (names : names) : prover =
  let numbering = names.numbering in
  { prover = "reify";
    of_compiler =
      (fun env input ->
        let among variables var =
          List.mem (Converter.variable_to_model numbering var) variables
        in
        A.Of_compiler.reification numbering
          (Reify.reify
             ~allowed_if_free_vars_defined_in:
               (match input.arguments.allowed_in with
               | The_environment -> env
               | Fewer_variables -> names.fewer_in_compiler)
             ~var_is_defined_at_toplevel:(among input.arguments.at_toplevel)
             ~var_is_symbol_projection:
               (among input.arguments.symbol_projections)
             env
             (Lazy.force input.asked_of_compiler)));
    of_model =
      (fun env input ->
        let among variables x = List.mem x variables in
        A.Of_model.reification
          (MR.reify
             (match input.arguments.allowed_in with
             | The_environment -> env
             | Fewer_variables -> names.fewer_in_model)
             (among input.arguments.at_toplevel)
             (among input.arguments.symbol_projections)
             env input.asked))
  }

(* The answer of the compiler, where it may stop: with a fatal error, or with an
   exception that a prover does not catch. The conversion of the arguments and
   of the answer runs inside the same guard, so an exception of these that the
   conversion raised would be taken for a stop of the compiler: that a caught
   stop comes from the function of the compiler is trusted. *)
let answer_of_compiler (prover : prover) (env : TE.t) (input : input) : A.answer
    =
  match
    Misc.Colours.without_colours ~f:(fun () ->
        or_fatal_error (fun () -> prover.of_compiler env input))
  with
  | Ok answer -> answer
  | Error line -> Stops ("a fatal error: " ^ general line)
  | exception ((Not_found | Invalid_argument _ | Assert_failure _) as exn) ->
    Stops ("an exception: " ^ Printexc.to_string exn)

(* The provers of provers.mli that take only naked numbers. *)
let is_on_naked_numbers (prover : prover) : bool =
  let prefix = "meet_naked_" in
  let length = String.length prefix in
  String.length prover.prover >= length
  && String.equal (String.sub prover.prover 0 length) prefix

(* ======================================================================== *)
(* The comparisons                                                          *)
(* ======================================================================== *)

(* Whether an environment of the compiler is seen the same as the model's, for
   the count: where it is not, no prover is asked in it. *)
let environment_is_seen_the_same (implementation : implementation)
    (of_environments : environments -> compiler_env) : input comparison =
  { name =
      Printf.sprintf
        "the environment made with the %s meet, seen the same in the model and \
         in the compiler"
        implementation.implementation;
    compare =
      (fun input ->
        match of_environments (Lazy.force input.environments) with
        | Seen_the_same (_ : TE.t) -> Agree
        | Not_asked why -> Not_run why)
  }

(* The kind checks of the compiler, for one run. *)
let with_kind_checks (f : unit -> 'a) : 'a =
  let before = !Oxcaml_flags.Flambda2.kind_checks in
  Oxcaml_flags.Flambda2.kind_checks := Oxcaml_flags.Set true;
  Fun.protect f ~finally:(fun () -> Oxcaml_flags.Flambda2.kind_checks := before)

let stops_at_a_kind_check (answer : A.answer) : bool =
  match answer with
  | Stops why ->
    let prefix = "a fatal error: Kind error" in
    let length = String.length prefix in
    String.length why >= length && String.equal (String.sub why 0 length) prefix
  | Proved _ | Unknown | Known_result _ | Need_meet | Invalid | Lift _
  | Simple _ | Cannot_reify ->
    false

(* A prover, asked in each environment of the compiler that is seen the same as
   the model's; a case in which neither is does not count, the two comparisons
   of the environments having counted it. With [kind_checks], the compiler is
   asked with its kind checks on: where the two answers then differ and the stop
   of the compiler has the message of a kind check, the difference is counted as
   the recorded one. That the model's answer is then the compiler's without the
   checks is what FlambdaProvers.v says; it is not asked here. *)
let comparison_of ?(kind_checks = false) (prover : prover) : input comparison =
  let answer_of_compiler prover env input =
    if kind_checks
    then with_kind_checks (fun () -> answer_of_compiler prover env input)
    else answer_of_compiler prover env input
  in
  { name = prover.prover;
    compare =
      (fun input ->
        let environments = Lazy.force input.environments in
        match environments.model with
        | Error (_ : string) -> Not_this_comparison
        | Ok model_env -> (
          let of_model = prover.of_model model_env input in
          let asked_in (implementation : implementation)
              (compiler_env : compiler_env) : outcome option =
            match compiler_env with
            | Not_asked (_ : string) -> None
            | Seen_the_same env ->
              let of_compiler = answer_of_compiler prover env input in
              let what () =
                Format.asprintf
                  "@[<v>the model answers %a,@ the compiler, in the \
                   environment made with the %s meet, %a@]"
                  A.print of_model implementation.implementation A.print
                  of_compiler
              in
              Some
                (if A.same of_model of_compiler
                 then Agree
                 else if kind_checks && stops_at_a_kind_check of_compiler
                 then
                   Known_to_disagree
                     ( "C2-2, with the kind checks on: the compiler stops on a \
                        type of another kind than the prover is for, and the \
                        model answers the default of the compiler without them",
                       what () )
                 else Disagree (what ()))
          in
          match
            ( asked_in n_way environments.with_n_way,
              asked_in binary environments.with_binary )
          with
          | None, None -> Not_this_comparison
          | Some outcome, None | None, Some outcome -> outcome
          | Some with_n_way, Some with_binary ->
            (* A disagreement in either environment is the outcome. *)
            let rank (outcome : outcome) =
              match outcome with
              | Disagree _ -> 2
              | Known_to_disagree _ -> 1
              | Agree | Not_run _ | Not_this_comparison -> 0
            in
            if rank with_n_way >= rank with_binary
            then with_n_way
            else with_binary))
  }

(* How the compiler answers, a prover at a time, on the inputs of a run: in the
   environment made with the n-way meet, where it is seen the same as the
   model's. This says what the comparisons above were made on. *)
let print_answers ?(kind_checks = false) ~(of_reification : bool) ~seed ~count
    ~generate (provers : prover list) =
  let answer_of_compiler prover env input =
    if kind_checks
    then with_kind_checks (fun () -> answer_of_compiler prover env input)
    else answer_of_compiler prover env input
  in
  let inputs =
    List.filter_map
      (fun i ->
        let input = generate (Random.of_int (seed + i)) in
        match prepare input with
        | input -> (
          match (Lazy.force input.environments).with_n_way with
          | Seen_the_same env -> Some (env, input)
          | Not_asked (_ : string) -> None)
        | exception
            ( Converter.Not_in_the_model _ | Converter.Not_in_the_compiler _
            | Misc.Fatal_error ) ->
          None)
      (List.init count Fun.id)
  in
  let columns =
    if of_reification
    then "Lift, then Simple, Cannot_reify, Invalid, no answer"
    else "a proof or a known result, Unknown or Need_meet, Invalid, no answer"
  in
  Format.printf
    "  The answers of the compiler in the environment made with the n-way meet \
     (%d cases):@.  %s@."
    (List.length inputs) columns;
  List.iter
    (fun prover ->
      let proved = ref 0
      and lifted = ref 0
      and simple = ref 0
      and unknown = ref 0
      and invalid = ref 0
      and stops = ref [] in
      List.iter
        (fun (env, input) ->
          match answer_of_compiler prover env input with
          | Proved _ | Known_result _ -> incr proved
          | Lift _ -> incr lifted
          | Simple _ -> incr simple
          | Unknown | Need_meet | Cannot_reify -> incr unknown
          | Invalid -> incr invalid
          | Stops why -> stops := why :: !stops
          | exception Converter.Not_in_the_model what ->
            stops := ("an answer that is not in the model: " ^ what) :: !stops)
        inputs;
      if of_reification
      then
        Format.printf "    %s: %d, then %d, %d, %d, %d@." prover.prover !lifted
          !simple !unknown !invalid (List.length !stops)
      else
        Format.printf "    %s: %d, %d, %d, %d@." prover.prover !proved !unknown
          !invalid (List.length !stops);
      List.iter
        (fun (reason, n) ->
          Format.printf "      %d with no answer: %s@." n reason)
        (count_reasons !stops))
    provers

(* What the comparisons can see, on the inputs of a run: a control of the tool,
   not of the provers.

   The answers of two provers that are not the same prover, one of the compiler
   and the other of the model, are compared as the answers of one prover are:
   they are to differ on some inputs.

   The environment of the compiler without the equations is compared with the
   model's with them, as the two environments of a case are: they are to be seen
   differently where an equation says something. *)
let controls (names : names) ~seed ~count ~generate (provers : prover list) =
  let inputs =
    List.filter_map
      (fun i ->
        let input = generate (Random.of_int (seed + i)) in
        match prepare input with
        | input -> (
          let environments = Lazy.force input.environments in
          match environments.model, environments.with_n_way with
          | Ok model_env, Seen_the_same env -> Some (model_env, env, input)
          | (Ok _ | Error _), (Seen_the_same _ | Not_asked _) -> None)
        | exception
            ( Converter.Not_in_the_model _ | Converter.Not_in_the_compiler _
            | Misc.Fatal_error ) ->
          None)
      (List.init count Fun.id)
  in
  let count_of condition = List.length (List.filter condition inputs) in
  Format.printf "  Controls, on the %d cases of the table above:@."
    (List.length inputs);
  let prover name =
    List.find (fun prover -> String.equal prover.prover name) provers
  in
  List.iter
    (fun (of_compiler, of_model) ->
      Format.printf
        "    %s of the compiler against %s of the model: the answers differ on \
         %d@."
        of_compiler of_model
        (count_of (fun (model_env, env, input) ->
             not
               (A.same
                  ((prover of_model).of_model model_env input)
                  (answer_of_compiler (prover of_compiler) env input)))))
    [ "prove_is_int", "prove_is_not_a_pointer";
      "prove_is_a_boxed_float", "prove_is_a_boxed_int64";
      "meet_boxed_float_containing_simple", "meet_boxed_mask_containing_simple";
      "meet_single_closures_entry", "meet_single_closures_entry" ];
  let type_is_known ((_ : MT.name), t) = not (MT.is_obviously_unknown t) in
  Format.printf
    "    %d have an equation whose type is not unknown; the environment of the \
     compiler without the equations is seen differently from the model's with \
     them on %d@."
    (count_of (fun (_, _, input) -> List.exists type_is_known input.equations))
    (count_of (fun (model_env, _, (_ : input)) ->
         Option.is_some (seen_differently names model_env names.compiler_env)));
  Format.printf
    "    in %d the type asked is an alias of a name that has such an equation@."
    (count_of (fun (_, _, input) ->
         match MT.get_alias_opt input.asked with
         | Some (Var x) ->
           List.exists
             (fun ((name : MT.name), t) ->
               MT.name_eqb name (Name_var x) && type_is_known (name, t))
             input.equations
         | Some (Symbol symbol) ->
           List.exists
             (fun ((name : MT.name), t) ->
               MT.name_eqb name (Name_symbol symbol) && type_is_known (name, t))
             input.equations
         | Some (Const _) | None -> false))

(* A case that no type of the model gives: two poison constants of one kind that
   have two names. A poison constant of the model has a kind and no name (the
   converter drops the name), so the two are one constant in the model and two
   in the compiler, for which constants are equal if they have one name too.
   [prove_physical_equality] of the aliases of the two is then [Proved false] in
   the compiler and [Proved true] in the model, as FlambdaProvers.v says: P1-1
   is the entry of what the compiler does of [Phys_equal] on poison. The types
   are made as the compiler's, and converted. *)
let poisons_of_two_names (names : names) =
  let poison name =
    TG.alias_type_of K.value
      (Simple.const (Reg_width_const.const_poison K.value name))
  in
  let asked label t1 t2 ~recorded =
    let of_compiler =
      A.Of_compiler.proof A.Of_compiler.bool
        (P.prove_physical_equality names.compiler_env t1 t2)
    in
    let to_model = Converter.type_to_model names.numbering in
    let of_model =
      A.Of_model.proof A.Of_model.bool
        (MP.prove_physical_equality names.model_env (to_model t1) (to_model t2))
    in
    let same = A.same of_model of_compiler in
    Format.printf "  %s: the model answers %a, the compiler %a%s@." label
      A.print of_model A.print of_compiler
      (if same
       then ""
       else if recorded
       then " (as recorded: P1-1)"
       else " (a DISAGREEMENT)");
    if (not same) && not recorded then incr failures
  in
  Format.printf
    "@.prove_physical_equality of the aliases of two poison constants of kind \
     value, which no type of the model gives@.";
  asked "of one name" (poison "p") (poison "p") ~recorded:false;
  asked "of two names" (poison "p") (poison "q") ~recorded:true

(* ======================================================================== *)
(* The tests                                                                *)
(* ======================================================================== *)

(* A run: its inputs, made from its seeds, and the provers asked on them. *)
type a_run =
  { title : string;
    seed : int;
    count : int;
    generate : Random.t -> input;
    shown : shown;
    (* The compiler is asked with its kind checks on. *)
    kind_checks : bool;
    asked : prover list;
    (* The controls are run on the inputs of this run. *)
    with_controls : bool
  }

let the_runs (names : names) : a_run list =
  let provers = provers names in
  let reification = [reification names] in
  let named names_of_provers =
    List.filter (fun prover -> List.mem prover.prover names_of_provers) provers
  in
  let one_type =
    List.filter
      (fun prover -> not (String.equal prover.prover "prove_physical_equality"))
      provers
  in
  let a_run ?(two_types = false) ?(of_reification = false)
      ?(kind_checks = false) ?(with_controls = false) title ~seed ~count
      generate asked =
    { title;
      seed;
      count;
      generate;
      shown = { two_types; of_reification };
      kind_checks;
      asked;
      with_controls
    }
  in
  [ a_run "A type of kind value, most of the time, in an environment"
      ~seed:100000 ~count:20000 (generate_on_values names) one_type
      ~with_controls:true;
    a_run "A type of naked numbers, most of the time, in an environment"
      ~seed:200000 ~count:6000
      (generate_on_naked_numbers names)
      (List.filter is_on_naked_numbers provers
      @ named
          [ "prove_equals_to_simple_of_kind";
            "prove_nothing";
            "never_holds_locally_allocated_values" ]);
    a_run
      "A type of naked immediates with relations to names, or the tagging of \
       one, in an environment"
      ~seed:250000 ~count:6000
      (generate_on_relations names)
      one_type;
    a_run "Two types, in an environment" ~seed:300000 ~count:20000
      (generate_two_types names) ~two_types:true
      (named ["prove_physical_equality"]);
    a_run "A raw type, and one more, in an environment" ~seed:400000
      ~count:20000 (generate_raw names) ~two_types:true provers;
    a_run
      "A type of kind value, most of the time, in an environment where a name \
       may have two equations"
      ~seed:500000 ~count:10000
      (generate_on_values ~twice:true names)
      one_type;
    a_run
      "Reification of a type of kind value, most of the time, in an environment"
      ~seed:600000 ~count:20000 (generate_on_values names) ~of_reification:true
      reification;
    a_run "Reification of a type of naked numbers, in an environment"
      ~seed:700000 ~count:4000
      (generate_on_naked_numbers names)
      ~of_reification:true reification;
    a_run "Reification of a raw type, in an environment" ~seed:800000
      ~count:10000 (generate_raw names) ~of_reification:true reification;
    a_run
      "Reification of a type of kind value, most of the time, in an \
       environment where a name may have two equations"
      ~seed:900000 ~count:6000
      (generate_on_values ~twice:true names)
      ~of_reification:true reification;
    a_run
      "With the kind checks of the compiler on: a type of any kind, and one \
       more, in an environment"
      ~seed:1000000 ~count:3000
      (generate_on_any_kind names)
      ~two_types:true ~kind_checks:true
      (named
         [ "prove_equals_to_simple_of_kind";
           "meet_naked_immediates";
           "meet_naked_floats";
           "prove_is_int";
           "prove_is_not_a_pointer";
           "meet_is_null";
           "meet_boxed_float_containing_simple";
           "never_holds_locally_allocated_values";
           "prove_physical_equality" ]) ]

(* One case, from its seed, which says of which run it is: the input, and the
   two answers of each prover of that run. For looking at a case by itself
   ([provers_difftest.exe -show SEED]). *)
let show (runs : a_run list) (seed : int) =
  match
    List.find_opt
      (fun (a_run : a_run) ->
        a_run.seed <= seed && seed < a_run.seed + a_run.count)
      runs
  with
  | None -> Format.printf "No run has the seed %d.@." seed
  | Some a_run -> (
    let input = prepare (a_run.generate (Random.of_int seed)) in
    Format.printf "@[<v>%s, seed %d:@ %a@]@." a_run.title seed
      (print_input a_run.shown) input;
    let environments = Lazy.force input.environments in
    let in_the_compiler (implementation : implementation) (env : compiler_env) =
      match env with
      | Not_asked why ->
        Format.printf "The environment made with the %s meet: %s@."
          implementation.implementation why
      | Seen_the_same (_ : TE.t) ->
        Format.printf
          "The environment made with the %s meet is seen the same as the \
           model's.@."
          implementation.implementation
    in
    in_the_compiler n_way environments.with_n_way;
    in_the_compiler binary environments.with_binary;
    let answer_of_compiler prover env =
      if a_run.kind_checks
      then with_kind_checks (fun () -> answer_of_compiler prover env input)
      else answer_of_compiler prover env input
    in
    match environments.model, environments.with_n_way with
    | Ok model_env, Seen_the_same env ->
      List.iter
        (fun prover ->
          Format.printf "  @[<v>%s:@ the model: %a@ the compiler: %a@]@."
            prover.prover A.print
            (prover.of_model model_env input)
            A.print
            (answer_of_compiler prover env))
        a_run.asked
    | (Ok _ | Error _), (Seen_the_same _ | Not_asked _) -> ())

let () =
  let comp_unit = Compilation_unit.of_string "Provers_difftest" in
  Env.set_current_unit
    (Unit_info.make_dummy ~input_name:"provers_difftest" comp_unit);
  let names = create_names () in
  let runs = the_runs names in
  (match Array.to_list Sys.argv with
  | [_; "-show"; seed] ->
    show runs (int_of_string seed);
    exit 0
  | _ -> ());
  Format.printf
    "The provers of the compiler against the Rocq model.@.Stack allocation: \
     %s.@.Kind checks: %s, but in the run that says so.@.Provers compared: %d, \
     and reification.@."
    (if Flambda2_ui.Flambda_features.stack_allocation_enabled ()
     then "enabled"
     else "disabled")
    (if Flambda2_ui.Flambda_features.kind_checks () then "on" else "off")
    (List.length (provers names));
  let environments =
    [ environment_is_seen_the_same n_way (fun environments ->
          environments.with_n_way);
      environment_is_seen_the_same binary (fun environments ->
          environments.with_binary) ]
  in
  List.iter
    (fun { title;
           seed;
           count;
           generate;
           shown;
           kind_checks;
           asked;
           with_controls
         } ->
      run ~title ~seed ~count ~generate ~prepare ~shrink:(shrink_input names)
        ~print:(print_input shown)
        (environments @ List.map (comparison_of ~kind_checks) asked);
      print_answers ~kind_checks ~of_reification:shown.of_reification ~seed
        ~count ~generate asked;
      if with_controls then controls names ~seed ~count ~generate asked)
    runs;
  poisons_of_two_names names;
  Format.printf "@.%s@."
    (if !failures = 0
     then "No disagreement."
     else Printf.sprintf "%d DISAGREEMENTS." !failures);
  if !failures > 0 then exit 1
