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

(* The provers and reification, asked in the environments that Simplify
   computes, replayed into the Rocq model.

   provers_difftest.ml asks the provers of the compiler and of the model in
   environments that it makes from generated equations. Here the environments
   are Simplify's own: a unit is read from a .fl file and simplified, and every
   typing environment that Simplify reports to the observer of the types library
   (at the start of each continuation handler that it simplifies, and at the end
   of the unit) is kept, and gone through when Simplify has returned. The
   observer is the hook of decision D33 ([Typing_env.For_testing.set_observer]);
   provers_of_compilations.README.md says what this test owes to it, and what a
   reported environment is and is not.

   There is no conversion of environments, and none is written here. An
   environment of Simplify, E, is REPLAYED: its names are defined again, in the
   order that [TE.stable_compare_simples] gives them (the symbols, then the
   variables in the order in which E bound them), each with its kind and with
   the name mode at which the environment has it; each name that is not
   canonical is made an alias of its canonical element; and each name is given
   the type that the environment has for it. That is done twice: in the model
   (M), with the types taken there by the converter; and in a new environment of
   the compiler (E'), with the compiler's own types. The steps are the same but
   for one: in E' a symbol of another unit is not defined, as the compiler has
   such a symbol without a definition; in M it is defined like any symbol. The
   steps are the functions of an environment that define a name, add an alias
   and replace a type; no equation is added and nothing is met, on either side.

   A replay is used only where it is seen the same as Simplify's environment
   ([first_difference] below): for each name, the answer of each function by
   which provers.ml and reify.ml read an environment, at each name mode. What is
   lost on the way is counted, each loss with its reason, before anything is
   asked in an environment. M is compared with E, which is the admission; E'
   with E; and E' with M, which says, where a replay does not give E back,
   whether the model's replay and the compiler's differ from each other.

   In an environment that is admitted, with E' seen as E too, the provers of the
   table (prover_table.ml) and reification are asked about the names, in E, in
   E' and in M, and the three answers are compared ([questions], [ask] below): a
   name is asked about with up to three inputs, and one prover is asked about
   variables only ([provers_for]). Where the two answers of the compiler agree
   and the model's is another, the case is shown and the run fails: it is a
   candidate, to be looked into, and not yet a fault of a prover. Where the two
   answers of the compiler differ, the replay has not given the compiler's
   answer back, and the run fails otherwise. An answer of the compiler that the
   model has no value for is counted as not compared, and is neither. A case of
   either kind is shown with its file, the flags, the environment, the question,
   the three answers and the names of the environment, each with what the
   replays are made from ([print_names]). Nothing is written that could be run
   without Simplify: to have a case again, the file is simplified again.

   What is printed for a file is a line: its environments, those that the
   model's replay is admitted for, those of them for which the compiler's replay
   is seen as E too, and the questions asked; and at the end, for all the files,
   the counts of the losses by reason and, for each prover, the answers on which
   the three agree and how the compiler answers. With [-why] a file is gone
   through alone and each environment that is not admitted with both replays is
   shown too. With [-control], an environment made by hand is gone through
   ([control]); with [-control-of-the-statuses], the tool runs itself on a file
   with answers spoiled, and says how each run ends ([control_of_the_statuses]).

   The status with which a check that returns ends. Of one file, in each way of
   going through one: 1 where a question is a candidate; else 4 where the
   compiler answers otherwise in its replay; else 0. Of a run over files: 1
   where a file has a candidate; else 4 where one has a failure of the replay;
   else 3 where the run on a file stopped; else 0. A file, or a run over files,
   that has more than one of these ends with the first of them, and the counts
   have each. A run on one file that stops before its check returns, at an
   exception that nothing catches, ends as the system makes it end; the run over
   files counts it as a stop.

   With [-O3] before the files, Simplify runs with the settings of that flag.
   Nothing else is set: the environments are those of the default flags. *)

open Flambda2_identifiers
open Flambda2_kinds
open Flambda2_parser
open Flambda2_term_basics
open Flambda2_terms
open Flambda2_bound_identifiers
open Flambda2_nominal
open Type_comparisons
module K = Flambda_kind
module TG = Flambda2_types__Type_grammar
module TE = Flambda2_types__Typing_env
module TEL = Flambda2_types__Typing_env_level
module EH = Flambda2_types__Expand_head
module Aliases = Flambda2_types__Aliases
module M = Flambda2_rocq_semantics.FlambdaModel
module MT = Flambda2_rocq_semantics.FlambdaTypeGrammar
module Model_env = Flambda2_rocq_semantics.FlambdaTypingEnv.Typing_env
module Model_expand = Flambda2_rocq_semantics.FlambdaExpandHead
module Model_name_mode = Flambda2_rocq_semantics.FlambdaBindingTime.Name_mode
module Converter = Flambda2_rocq_types_converter.Type_converter
module Model_names = Flambda2_rocq_types_converter.Model_names
module Model_type = Flambda2_rocq_types_converter.Model_type
module N = Flambda2_rocq_types_converter.Model_numbers
module A = Flambda2_rocq_types_converter.Prover_answers
module PT = Prover_table

(* ======================================================================== *)
(* The names of an environment                                              *)
(* ======================================================================== *)

type mode =
  | Normal
  | In_types
  | Phantom

(* The three name modes, the strongest first: a name that is there at a mode is
   there at every mode after it ([Name_mode.compare_partial_order]). *)
let modes = [Normal; Phantom; In_types]

let mode_of_compiler (mode : mode) : Name_mode.t =
  match mode with
  | Normal -> Name_mode.normal
  | In_types -> Name_mode.in_types
  | Phantom -> Name_mode.phantom

let mode_of_model (mode : mode) : Model_name_mode.t =
  match mode with Normal -> Normal | In_types -> In_types | Phantom -> Phantom

let print_mode (mode : mode) =
  match mode with
  | Normal -> "normal"
  | In_types -> "in types"
  | Phantom -> "phantom"

(* A name that an environment has: its kind, and the name mode at which the
   environment has it, which is the strongest of the three at which [TE.mem]
   says that the name is there ([modes] is gone through from the strongest). For
   a variable that was bound outside the closure that Simplify is in, that is
   the mode for types, whatever the mode it was bound with: the environment
   gives it no other. *)
type name_of_env =
  { name : Name.t;
    kind : K.t;
    mode : mode
  }

exception Lost of string

let lost fmt = Format.kasprintf (fun why -> raise (Lost why)) fmt

let kind_of_name (name : Name.t) : K.t =
  Name.pattern_match name ~var:Variable.kind ~symbol:(fun _ -> K.value)

let mode_in (env : TE.t) (name : Name.t) : mode option =
  List.find_opt
    (fun mode -> TE.mem ~min_name_mode:(mode_of_compiler mode) env name)
    modes

(* The names of an environment: the variables that its levels define and the
   symbols that it defines, as in types_of_compilations.ml; and with them every
   name that the type of one of these mentions, and every name that the
   environment has as an alias of one, and so on: a type may mention a name of
   another unit, which no level defines. A name that a type mentions and the
   environment has not is a loss. The variables are in the order in which the
   environment bound them, after the symbols. *)
let names_of (env : TE.t) : name_of_env list =
  let level = TE.cut env ~cut_after:Scope.initial in
  let defined =
    List.map
      (fun (variable, (_ : K.t)) -> Name.var variable)
      (Variable.Map.bindings (TEL.defined_variables_with_kinds level))
    @ List.map Name.symbol (Symbol.Set.elements (TE.defined_symbols env))
  in
  let rec close (seen : Name.Set.t) (found : name_of_env list)
      (names : Name.t list) : name_of_env list =
    match names with
    | [] -> found
    | name :: names -> (
      if Name.Set.mem name seen
      then close seen found names
      else
        let seen = Name.Set.add name seen in
        match mode_in env name with
        | None ->
          lost "a type mentions a %s that the environment has not"
            (Name.pattern_match name
               ~var:(fun _ -> "variable")
               ~symbol:(fun _ -> "symbol"))
        | Some mode ->
          let kind = kind_of_name name in
          let mentioned =
            Name_occurrences.fold_names
              (TG.free_names (TE.find env name (Some kind)))
              ~init:[]
              ~f:(fun names name -> name :: names)
          in
          (* The aliases of the name, at the two least name modes: the set is
             gone through by a filter that keeps all of it. *)
          let aliases =
            List.concat_map
              (fun min_name_mode ->
                let aliases = ref [] in
                let simple = Simple.name name in
                ignore
                  (Aliases.Alias_set.filter
                     (TE.aliases_of_simple env ~min_name_mode simple)
                     ~f:(fun alias ->
                       Simple.pattern_match alias
                         ~name:(fun name ~coercion:_ ->
                           aliases := name :: !aliases)
                         ~const:(fun _ -> ());
                       true)
                    : Aliases.Alias_set.t);
                !aliases)
              [Name_mode.in_types; Name_mode.phantom]
          in
          close seen
            ({ name; kind; mode } :: found)
            (mentioned @ aliases @ names))
  in
  let names = close Name.Set.empty [] defined in
  List.sort
    (fun { name; _ } { name = name'; _ } ->
      TE.stable_compare_simples env (Simple.name name) (Simple.name name'))
    names

(* ======================================================================== *)
(* The replay                                                               *)
(* ======================================================================== *)

(* What Simplify's environment has for a name: its type, as [find] gives it, and
   its canonical element whatever the name mode.

   The two are not one thing. The type of a name that is not canonical is not
   always the alias of its canonical element: where the environment has [x : (=
   y)] and [y : (= z)], the type of [x] was stored when [y] was still canonical
   and nothing rewrote it; and a join gives an environment aliases that it
   records beside the types and not in them, so that two names are aliases and
   each keeps a type that is no alias. So a replay does not add equations, which
   would meet and store other types: it adds the aliases, each name that is not
   canonical with its canonical element, and then puts the type of each name
   where it was, by the function of the environment that replaces a type.
   Whether that gives the environment back is for the gate to say. *)
type of_simplify =
  { name_of_env : name_of_env;
    t : TG.t;
    canonical : Simple.t
  }

let read (env : TE.t) (names : name_of_env list) : of_simplify list =
  List.map
    (fun ({ name; kind; mode = _ } as name_of_env) ->
      { name_of_env;
        t = TE.find env name (Some kind);
        canonical =
          TE.get_canonical_simple_ignoring_name_mode env (Simple.name name)
      })
    names

(* The same, in the model. A type or a canonical element that the model has not
   is a loss. *)
type in_the_model =
  { of_simplify : of_simplify;
    t_in_model : MT.flambda_type;
    canonical_in_model : M.simple
  }

let to_the_model (numbering : Model_names.t) (read : of_simplify list) :
    in_the_model list =
  List.map
    (fun ({ t; canonical; name_of_env = _ } as of_simplify) ->
      match
        ( Converter.type_to_model numbering t,
          Converter.simple_to_model numbering canonical )
      with
      | t_in_model, canonical_in_model ->
        { of_simplify; t_in_model; canonical_in_model }
      | exception Converter.Not_in_the_model what ->
        lost "a type that the model has not: %s" (general what))
    read

let is_canonical ({ name_of_env = { name; _ }; canonical; t = _ } : of_simplify)
    : bool =
  Simple.equal canonical (Simple.name name)

(* The environment of the model: the names defined in their order; then each
   name that is not canonical made an alias of its canonical element; then the
   type of each name put in its place, but the unknown type, which a name has
   from its definition. *)
let replay_in_the_model (numbering : Model_names.t) (read : in_the_model list) :
    Model_env.t =
  let env =
    List.fold_left
      (fun env { of_simplify = { name_of_env = { name; kind; mode }; _ }; _ } ->
        Name.pattern_match name
          ~var:(fun variable ->
            Model_env.add_variable_definition env
              (Converter.variable_to_model numbering variable)
              (Converter.kind_to_model kind)
              (mode_of_model mode))
          ~symbol:(fun symbol ->
            Model_env.add_symbol_definition env
              (Converter.symbol_to_model numbering symbol)))
      Model_env.create read
  in
  let env =
    List.fold_left
      (fun env { of_simplify; canonical_in_model; t_in_model = _ } ->
        if is_canonical of_simplify
        then env
        else
          let name = of_simplify.name_of_env.name in
          match
            Model_env.add_alias env canonical_in_model
              (MT.simple_of_name (Converter.name_to_model numbering name))
          with
          | Some (Ok result) -> Model_env.add_alias_result_t result
          | Some (Unknown | Bottom) ->
            lost "the model does not add an alias (unknown, or bottom)"
          | None -> lost "the model has no answer where an alias is added")
      env read
  in
  List.fold_left
    (fun env
         { of_simplify = { name_of_env = { name; _ }; t; _ }; t_in_model; _ } ->
      if TG.is_obviously_unknown t
      then env
      else
        Model_env.replace_equation env
          (Converter.name_to_model numbering name)
          t_in_model)
    env read

(* A new environment of the compiler, by the same steps, with the compiler's own
   types. A symbol of another unit is not defined: the compiler has it without a
   definition. *)
let replay_in_the_compiler (read : of_simplify list) : (TE.t, string) result =
  match
    Misc.Colours.without_colours ~f:(fun () ->
        or_fatal_error (fun () ->
            let env =
              List.fold_left
                (fun env { name_of_env = { name; kind; mode }; _ } ->
                  Name.pattern_match name
                    ~var:(fun variable ->
                      TE.add_definition env
                        (Bound_name.create_var
                           (Bound_var.create variable Flambda_debug_uid.none
                              (mode_of_compiler mode)))
                        kind)
                    ~symbol:(fun symbol ->
                      if
                        Current_unit.is_current (Symbol.compilation_unit symbol)
                      then
                        TE.add_definition env
                          (Bound_name.create_symbol symbol)
                          K.value
                      else env))
                (TE.create ~machine_width ~resolver:(fun _ -> None))
                read
            in
            let env =
              List.fold_left
                (fun env ({ name_of_env = { name; _ }; canonical; _ } as read)
                   ->
                  if is_canonical read
                  then env
                  else
                    match
                      TE.add_alias env ~canonical_element1:canonical
                        ~canonical_element2:(Simple.name name)
                    with
                    | Ok { t = env; canonical_element = _; demoted_name = _ } ->
                      env
                    | Unknown | Bottom ->
                      lost
                        "the compiler does not add an alias (unknown, or \
                         bottom)")
                env read
            in
            List.fold_left
              (fun env { name_of_env = { name; _ }; t; _ } ->
                if TG.is_obviously_unknown t
                then env
                else TE.replace_equation env name t)
              env read))
  with
  | Ok env -> Ok env
  | Error line ->
    Error ("a fatal error where the compiler replays: " ^ general line)
  | exception Lost why -> Error why

(* ======================================================================== *)
(* What is seen of an environment                                           *)
(* ======================================================================== *)

(* What an environment answers of one name, by each function by which provers.ml
   and reify.ml read an environment: [find], [mem], [get_canonical_simple_exn],
   [get_canonical_simple_ignoring_name_mode],
   [get_alias_then_canonical_simple_exn] and [expand_head] (found by reading the
   two files; the functions that these call in turn are not listed here). They
   read [machine_width] too: that is checked where an environment is replayed,
   and the model has none.

   The answers are values of the model, so that an environment of the compiler
   is compared in one way with one of the model and with another of the
   compiler: a type is compared in the normal form of model_type.ml. A canonical
   element that there is not ([Not_found] in the compiler) is [None]. *)
type seen =
  { found : MT.flambda_type;  (** [find] with the kind of the name. *)
    found_without_kind : MT.flambda_type;
    there : bool list;  (** At no name mode, then at each of [modes]. *)
    canonical : M.simple option list;  (** At each of [modes]. *)
    canonical_ignoring_mode : M.simple;
    alias_then_canonical : M.simple option list;
        (** Of the type found, at each of [modes]. *)
    expanded : MT.flambda_type  (** Of the alias of the name. *)
  }

let modes_or_none : mode option list = None :: List.map Option.some modes

(* What an environment of the compiler answers of a name. *)
let seen_in_the_compiler (numbering : Model_names.t) (env : TE.t)
    { name; kind; mode = _ } : seen =
  let to_model = Converter.type_to_model numbering in
  let simple_to_model = Converter.simple_to_model numbering in
  let simple = Simple.name name in
  let found = TE.find env name (Some kind) in
  let or_not_found f =
    match f () with
    | canonical -> Some (simple_to_model canonical)
    | exception Not_found -> None
  in
  { found = to_model found;
    found_without_kind = to_model (TE.find env name None);
    there =
      List.map
        (fun mode ->
          TE.mem ?min_name_mode:(Option.map mode_of_compiler mode) env name)
        modes_or_none;
    canonical =
      List.map
        (fun mode ->
          or_not_found (fun () ->
              TE.get_canonical_simple_exn env
                ~min_name_mode:(mode_of_compiler mode) simple))
        modes;
    canonical_ignoring_mode =
      simple_to_model (TE.get_canonical_simple_ignoring_name_mode env simple);
    alias_then_canonical =
      List.map
        (fun mode ->
          or_not_found (fun () ->
              TE.get_alias_then_canonical_simple_exn env
                ~min_name_mode:(mode_of_compiler mode) found))
        modes;
    expanded =
      to_model
        (EH.Expanded_type.to_type
           (EH.expand_head env (TG.alias_type_of kind simple)))
  }

(* A function of the model that has no answer: not the same thing as a canonical
   element that there is not. *)
exception The_model_has_no_answer of string

(* What the environment of the model answers of a name. *)
let seen_in_the_model (numbering : Model_names.t) (model_env : Model_env.t)
    { name; kind; mode = _ } : seen =
  let in_model = Converter.name_to_model numbering name in
  let simple = MT.simple_of_name in_model in
  let or_no_answer what (answer : _ option) =
    match answer with
    | Some answer -> answer
    | None -> raise (The_model_has_no_answer what)
  in
  let found =
    or_no_answer "find"
      (Model_env.find model_env in_model (Some (Converter.kind_to_model kind)))
  in
  { found;
    found_without_kind =
      or_no_answer "find" (Model_env.find model_env in_model None);
    there =
      List.map
        (fun mode ->
          Model_env.mem (Option.map mode_of_model mode) model_env in_model)
        modes_or_none;
    canonical =
      List.map
        (fun mode ->
          or_no_answer "get_canonical_simple_exn"
            (Model_env.get_canonical_simple_exn model_env
               (Some (mode_of_model mode))
               None simple))
        modes;
    canonical_ignoring_mode =
      Model_env.get_canonical_simple_ignoring_name_mode model_env simple;
    alias_then_canonical =
      List.map
        (fun mode ->
          or_no_answer "get_alias_then_canonical_simple_exn"
            (Model_env.get_alias_then_canonical_simple_exn model_env
               (Some (mode_of_model mode))
               None found))
        modes;
    expanded =
      Model_expand.Expanded_type.to_type
        (or_no_answer "expand_head"
           (Model_expand.expand_head model_env
              (MT.alias_type_of (Converter.kind_to_model kind) simple)))
  }

(* What is seen of every name of an environment, and whether it is bottom; or
   why it cannot be read. *)
type seen_of_env =
  { is_bottom : bool;
    of_names : seen list
  }

let read_the_compiler (numbering : Model_names.t) (names : name_of_env list)
    (env : TE.t) : (seen_of_env, string) result =
  match
    Misc.Colours.without_colours ~f:(fun () ->
        or_fatal_error (fun () ->
            { is_bottom = TE.is_bottom env;
              of_names = List.map (seen_in_the_compiler numbering env) names
            }))
  with
  | Ok seen -> Ok seen
  | Error line ->
    Error ("a fatal error where the environment is read: " ^ general line)
  | exception Converter.Not_in_the_model what ->
    Error
      ("what is read of the environment is not in the model: " ^ general what)

let read_the_model (numbering : Model_names.t) (names : name_of_env list)
    (model_env : Model_env.t) : (seen_of_env, string) result =
  match
    { is_bottom = Model_env.is_bottom model_env;
      of_names = List.map (seen_in_the_model numbering model_env) names
    }
  with
  | seen -> Ok seen
  | exception The_model_has_no_answer what ->
    Error ("the model has no answer where the environment is read: " ^ what)

(* The first thing that differs in what is seen of two environments: what it is,
   for the counts, and for a name which name and the two answers. *)
type difference =
  { what : string;
    shown : Format.formatter -> unit
  }

let first_difference (names : name_of_env list) (seen : seen_of_env)
    (seen' : seen_of_env) : difference option =
  let simple ppf (simple : M.simple option) =
    match simple with
    | None -> Format.pp_print_string ppf "none"
    | Some simple -> Model_type.print_simple ppf simple
  in
  let at what mode = what ^ ", at the name mode " ^ print_mode mode in
  let of_name ({ name; _ } : name_of_env) (a : seen) (b : seen) :
      difference option =
    let differs what print x y =
      Some
        { what;
          shown =
            (fun ppf ->
              Format.fprintf ppf "@[<v>of %a:@ %a@ against@ %a@]" Name.print
                name print x print y)
        }
    in
    let first_of_modes what print xs ys =
      List.find_map
        (fun (mode, (x, y)) ->
          if Stdlib.( = ) x y then None else differs (at what mode) print x y)
        (List.combine modes (List.combine xs ys))
    in
    if not (Model_type.same a.found b.found)
    then differs "the type of a name" Model_type.print a.found b.found
    else if not (Model_type.same a.found_without_kind b.found_without_kind)
    then
      differs "the type of a name, asked without its kind" Model_type.print
        a.found_without_kind b.found_without_kind
    else if not (List.equal Bool.equal a.there b.there)
    then
      differs "whether a name is there, at a name mode"
        (Model_type.print_list Format.pp_print_bool)
        a.there b.there
    else
      match
        first_of_modes "the canonical element of a name" simple a.canonical
          b.canonical
      with
      | Some _ as difference -> difference
      | None -> (
        if
          not (Stdlib.( = ) a.canonical_ignoring_mode b.canonical_ignoring_mode)
        then
          differs "the canonical element of a name, whatever its name mode"
            Model_type.print_simple a.canonical_ignoring_mode
            b.canonical_ignoring_mode
        else
          match
            first_of_modes
              "the canonical element of what the type of a name is an alias of"
              simple a.alias_then_canonical b.alias_then_canonical
          with
          | Some _ as difference -> difference
          | None ->
            if not (Model_type.same a.expanded b.expanded)
            then
              differs "the expanded head of a name" Model_type.print a.expanded
                b.expanded
            else None)
  in
  if not (Bool.equal seen.is_bottom seen'.is_bottom)
  then
    Some
      { what = "one is bottom";
        shown = (fun ppf -> Format.pp_print_string ppf "one is bottom")
      }
  else
    let rec first names (seen : seen list) (seen' : seen list) =
      match names, seen, seen' with
      | name :: names, a :: seen, b :: seen' -> (
        match of_name name a b with
        | Some _ as difference -> difference
        | None -> first names seen seen')
      | _, _, _ -> None
    in
    first names seen.of_names seen'.of_names

(* ======================================================================== *)
(* One environment                                                          *)
(* ======================================================================== *)

(* Two environments that are to be seen the same, and whether they are. *)
type compared =
  | Seen_the_same
  | Differ of difference
  | Not_compared of string  (** One of the two is not there, or not read. *)

let what_of (compared : compared) : string =
  match compared with
  | Seen_the_same -> "seen the same"
  | Differ { what; shown = _ } -> what
  | Not_compared why -> why

(* An environment of Simplify, E, with its two replays: the model's, M, and the
   compiler's, E'. *)
type replayed =
  { names : name_of_env list;
    read : in_the_model list;
    env : TE.t;
    model_env : (Model_env.t, string) result;
    replayed_env : (TE.t, string) result;
    simplify_against_model : compared;  (** E against M: the admission. *)
    simplify_against_replay : compared;  (** E against E'. *)
    replay_against_model : compared  (** E' against M. *)
  }

(* The replays of an environment, or why it is lost before any is made. *)
let replay (numbering : Model_names.t) (env : TE.t) : (replayed, string) result
    =
  match
    if
      not
        (Target_system.Machine_width.equal (TE.machine_width env) machine_width)
    then lost "the width of the machine is not the one of this test";
    let names =
      match
        Misc.Colours.without_colours ~f:(fun () ->
            or_fatal_error (fun () -> names_of env))
      with
      | Ok names -> names
      | Error line ->
        lost "a fatal error where the names are read: %s" (general line)
    in
    let read =
      match
        Misc.Colours.without_colours ~f:(fun () ->
            or_fatal_error (fun () -> to_the_model numbering (read env names)))
      with
      | Ok read -> read
      | Error line ->
        lost "a fatal error where the environment is read: %s" (general line)
    in
    let seen =
      match read_the_compiler numbering names env with
      | Ok seen -> seen
      | Error why -> lost "%s" why
    in
    let model_env =
      match replay_in_the_model numbering read with
      | model_env -> Ok model_env
      | exception Lost why -> Error why
    in
    let replayed_env =
      replay_in_the_compiler
        (List.map (fun { of_simplify; _ } -> of_simplify) read)
    in
    let seen_of_model =
      Result.bind model_env (read_the_model numbering names)
    in
    let seen_of_replay =
      Result.bind replayed_env (read_the_compiler numbering names)
    in
    let compare seen (seen' : (seen_of_env, string) result) : compared =
      match seen' with
      | Error why -> Not_compared why
      | Ok seen' -> (
        match first_difference names seen seen' with
        | None -> Seen_the_same
        | Some difference -> Differ difference)
    in
    { names;
      read;
      env;
      model_env;
      replayed_env;
      simplify_against_model = compare seen seen_of_model;
      simplify_against_replay = compare seen seen_of_replay;
      replay_against_model =
        (match seen_of_replay with
        | Error why -> Not_compared why
        | Ok seen_of_replay -> compare seen_of_replay seen_of_model)
    }
  with
  | replayed -> Ok replayed
  | exception Lost why -> Error why

(* ======================================================================== *)
(* The counts                                                               *)
(* ======================================================================== *)

(* Counts by what is counted, shown in the order in which each was first
   counted. *)
type counts =
  { mutable order : string list;
    table : (string, int) Hashtbl.t
  }

let create_counts () = { order = []; table = Hashtbl.create 16 }

let add counts what n =
  match Hashtbl.find_opt counts.table what with
  | Some m -> Hashtbl.replace counts.table what (m + n)
  | None ->
    counts.order <- what :: counts.order;
    Hashtbl.replace counts.table what n

let count counts what = add counts what 1

let number (counts : counts) what =
  Option.value (Hashtbl.find_opt counts.table what) ~default:0

(* What is counted of the answers of a prover has a key of its own: the name of
   the prover and what is counted, so that the counts of a prover are shown on
   one line. *)
let key_of_prover (prover : string) (what : string) : string =
  String.concat "|" ["asked"; prover; what]

let of_a_prover (key : string) : (string * string) option =
  match String.split_on_char '|' key with
  | "asked" :: prover :: what -> Some (prover, String.concat "|" what)
  | _ -> None

let agree = "agree"

let classes_of_answers =
  ["a proof or a result"; "unknown"; "invalid"; "no answer"]

let print_counts counts =
  let order = List.rev counts.order in
  List.iter
    (fun what ->
      if Option.is_none (of_a_prover what)
      then Format.printf "  %d %s@." (Hashtbl.find counts.table what) what)
    order;
  let provers =
    List.fold_left
      (fun provers key ->
        match of_a_prover key with
        | Some (prover, (_ : string)) when not (List.mem prover provers) ->
          prover :: provers
        | Some _ | None -> provers)
      [] order
    |> List.rev
  in
  if not (List.is_empty provers)
  then
    Format.printf
      "  The answers, a prover at a time: those on which the three agree; then \
       how the compiler answers in Simplify's environment: %s@."
      (String.concat ", " classes_of_answers);
  List.iter
    (fun prover ->
      let number what = number counts (key_of_prover prover what) in
      Format.printf "    %s: %d agree; %s@." prover (number agree)
        (String.concat ", "
           (List.map
              (fun what -> string_of_int (number what))
              classes_of_answers));
      List.iter
        (fun key ->
          match of_a_prover key with
          | Some (prover', what)
            when String.equal prover prover'
                 && (not (String.equal what agree))
                 && not (List.mem what classes_of_answers) ->
            Format.printf "      %d %s@." (Hashtbl.find counts.table key) what
          | Some _ | None -> ())
        order)
    provers

(* The counts of one file are written to a file, a count and what is counted on
   each line, for the run that goes through all the files. *)
let write_counts counts path =
  let channel = open_out path in
  List.iter
    (fun what ->
      Printf.fprintf channel "%d %s\n" (Hashtbl.find counts.table what) what)
    (List.rev counts.order);
  close_out channel

let add_counts totals path =
  let channel = open_in path in
  let rec read () =
    match input_line channel with
    | line ->
      (match String.index_opt line ' ' with
      | Some i -> (
        match int_of_string_opt (String.sub line 0 i) with
        | Some n ->
          add totals (String.sub line (i + 1) (String.length line - i - 1)) n
        | None -> ())
      | None -> ());
      read ()
    | exception End_of_file -> close_in channel
  in
  read ()

(* ======================================================================== *)
(* The questions                                                            *)
(* ======================================================================== *)

(* How a name is asked about: up to three inputs.

   Simplify asks a prover of the type that it has for a simple of a term
   ([Simplify_simple]: [TE.type_simple_in_term_exn]), which is the alias of the
   canonical simple at the name mode normal: that is the first input, for a name
   that the environment has at that mode and for which that function has an
   answer (where it raises [Not_found], the name is not asked about with this
   input). It is the input that Simplify would supply for that name there, not
   one that Simplify did supply: an environment is reported at the start of a
   handler, and no prover is called there.

   The second is the alias of the name itself, whatever its mode: types hold
   such aliases, and a prover is given them when it goes into a type.

   The third is the type that the environment has for the name ([TE.find]): a
   head, where the name is canonical, or an alias, which may be one that was
   stored before its target became an alias. Reification answers a simple for an
   alias of a name that is allowed; it is on this input that it has to say what
   a type holds. *)
type input_of =
  | As_simplify_simple
  | Raw_alias
  | Type_found

let print_input_of (input_of : input_of) =
  match input_of with
  | As_simplify_simple ->
    "the type that Simplify has for it in a term (the alias of its canonical \
     simple)"
  | Raw_alias -> "the alias of the name itself"
  | Type_found -> "the type that the environment has for it"

(* What does not change from one environment of a unit to another: the
   numbering, the table of the provers, and two slots that no type has. *)
type of_unit =
  { numbering : Model_names.t;
    provers : PT.prover list;
    no_function_slot : M.function_slot;
    no_value_slot : M.value_slot
  }

let of_unit () : of_unit =
  let numbering = Model_names.create () in
  let unit = Current_unit.get_cu_exn () in
  let kind_of_value_slot slot =
    Converter.kind_to_model
      (Value_slot.kind (Converter.value_slot_of_model numbering slot))
  in
  { numbering;
    provers = PT.provers ~numbering ~kind_of_value_slot;
    no_function_slot =
      numbering.function_slots.number
        (Function_slot.create unit ~name:"no_type_has_it" ~size:1);
    no_value_slot =
      numbering.value_slots.number
        (Value_slot.create unit ~name:"no_type_has_it"
           ~is_always_immediate:false K.value)
  }

(* A question about a name of an environment, with what says which it is. *)
type asked =
  { about : name_of_env;
    input_of : input_of;
    question : PT.question
  }

(* The first function slot and the first value slot of a type whose expanded
   head, in the model, is a type of closures. *)
let slots_of (model_env : Model_env.t) (t : MT.flambda_type) :
    M.function_slot option * M.value_slot option =
  let of_closures (Row_like_for_closures cases : MT.row_like_for_closures) =
    let entries =
      List.map
        (fun (_, (case : (_, _, _, _) MT.row_like_case)) ->
          let (Closures_entry (_, closure_types, value_slot_types)
                : MT.closures_entry) =
            case.case_maps_to
          in
          closure_types, value_slot_types)
        cases
    in
    let first slots =
      match List.concat_map slots entries with
      | (slot, (_ : MT.flambda_type)) :: _ -> Some slot
      | [] -> None
    in
    first fst, first snd
  in
  match
    Option.map Model_expand.Expanded_type.to_type
      (Model_expand.expand_head model_env t)
  with
  | None -> None, None
  | Some (Type_value (Ok (No_alias (Head_of_kind_value (non_null, _))))) -> (
    match non_null with
    | Ok (Closures (closures, _)) -> of_closures closures
    | Ok (Variant _ | Mutable_block _ | Boxed_number _ | String _)
    | Unknown | Bottom ->
      None, None)
  | Some
      ( Type_value (Unknown | Bottom | Ok (Equals _))
      | Type_naked_immediate _ | Type_naked_number _ | Type_rec_info _
      | Type_region _ ) ->
    None, None

(* The questions of an environment: for each name, its inputs, which are up to
   three. The other arguments are made by rules, the same for the three sides:

   - the least name mode of a simple that is answered: normal for the first
   input, the mode for types for the other two;

   - the kind: that of the name;

   - the index of a field: the place of the name among the names, modulo three;

   - the function slot and the value slot: the first of each in the type, where
   its expanded head is a type of closures, and otherwise a slot that no type
   has;

   - the variable (of [never_holds_locally_allocated_values], which is asked of
   a variable and of no type): the name, where it is a variable; that prover is
   not asked about a symbol;

   - the second type (of [prove_physical_equality]): the type of the question
   before of the same input and the same kind, or the type itself;

   - of reification: the variables that are allowed are those of the environment
   itself; the projections from symbols are the variables that Simplify's
   environment says are ([TE.find_symbol_projection]), given alike to the three
   sides; and the variables at toplevel, which an environment does not say, are
   all the variables here, and none where reification is asked a second time
   ([with_none_at_toplevel]). None of these is what Simplify gave where it
   reified. *)
let questions (of_unit : of_unit) (replayed : replayed)
    (model_env : Model_env.t) : asked list =
  let numbering = of_unit.numbering in
  let env = replayed.env in
  let variables =
    List.filter_map
      (fun { name; _ } ->
        Name.pattern_match name
          ~var:(fun variable -> Some variable)
          ~symbol:(fun _ -> None))
      replayed.names
  in
  let in_model = List.map (Converter.variable_to_model numbering) in
  let at_toplevel = in_model variables in
  let symbol_projections =
    in_model
      (List.filter
         (fun variable ->
           Option.is_some (TE.find_symbol_projection env variable))
         variables)
  in
  let before : (input_of * K.t, MT.flambda_type * TG.t) Hashtbl.t =
    Hashtbl.create 16
  in
  List.concat
    (List.mapi
       (fun i ({ name; kind; mode } as about) ->
         let ask input_of (t : TG.t) : asked list =
           let asked = Converter.type_to_model numbering t in
           let other, other_of_compiler =
             Option.value
               (Hashtbl.find_opt before (input_of, kind))
               ~default:(asked, t)
           in
           Hashtbl.replace before (input_of, kind) (asked, t);
           let function_slot, value_slot = slots_of model_env asked in
           let arguments : PT.arguments =
             { min_name_mode =
                 (match input_of with
                 | As_simplify_simple -> Normal
                 | Raw_alias | Type_found -> In_types);
               kind = Converter.kind_to_model kind;
               field_index = i mod 3;
               function_slot =
                 Option.value function_slot ~default:of_unit.no_function_slot;
               value_slot =
                 Option.value value_slot ~default:of_unit.no_value_slot;
               variable =
                 Name.pattern_match name
                   ~var:(Converter.variable_to_model numbering)
                   ~symbol:(fun _ -> N.positive_of_int 1);
               allowed_in = The_environment;
               at_toplevel;
               symbol_projections
             }
           in
           [ { about;
               input_of;
               question =
                 { asked;
                   other;
                   arguments;
                   asked_of_compiler = Lazy.from_val t;
                   other_of_compiler = Lazy.from_val other_of_compiler
                 }
             } ]
         in
         (match mode with
           | Normal -> (
             match
               TE.type_simple_in_term_exn env ~min_name_mode:Name_mode.normal
                 (Simple.name name)
             with
             | t, (_ : Simple.t) -> ask As_simplify_simple t
             | exception Not_found -> [])
           | In_types | Phantom -> [])
         @ ask Raw_alias (TG.alias_type_of kind (Simple.name name))
         @ ask Type_found (TE.find env name (Some kind)))
       replayed.names)

let with_none_at_toplevel (question : PT.question) : PT.question =
  { question with arguments = { question.arguments with at_toplevel = [] } }

(* The provers that are asked of a question: those of the table, and reification
   twice. *)
let provers_for (of_unit : of_unit) (replayed : replayed)
    (model_env : Model_env.t) : asked -> PT.prover list =
  let reification =
    PT.reification ~numbering:of_unit.numbering ~fewer_in_compiler:replayed.env
      ~fewer_in_model:model_env
  in
  let reifications : PT.prover list =
    [ { reification with prover = "reify, every variable at toplevel" };
      { prover = "reify, no variable at toplevel";
        of_compiler =
          (fun env question ->
            reification.of_compiler env (with_none_at_toplevel question));
        of_model =
          (fun env question ->
            reification.of_model env (with_none_at_toplevel question))
      } ]
  in
  let of_a_variable = "never_holds_locally_allocated_values" in
  fun { about = { name; _ }; _ } ->
    List.filter
      (fun (prover : PT.prover) ->
        Name.is_var name || not (String.equal prover.prover of_a_variable))
      of_unit.provers
    @ reifications

(* The three answers to a question: of the compiler in Simplify's environment,
   of the compiler in its replay, and of the model in its replay. *)
type answers =
  { in_simplify : A.answer;
    in_replay : A.answer;
    in_model : A.answer
  }

type verdict =
  | The_three_agree
  | The_compiler_differs_in_its_replay
      (** The replay does not give the compiler's answer back: a failure of this
          test, and no comparison with the model. *)
  | The_model_differs
      (** The two answers of the compiler agree and the model's is another: a
          candidate, to be looked into. The replay of the model, the converter
          and the conversion of the answers are on the way as well as the
          prover. *)

let verdict { in_simplify; in_replay; in_model } : verdict =
  if not (A.same in_simplify in_replay)
  then The_compiler_differs_in_its_replay
  else if A.same in_simplify in_model
  then The_three_agree
  else The_model_differs

let class_of_answer (answer : A.answer) : string =
  match answer with
  | Proved _ | Known_result _ | Lift _ | Simple _ -> "a proof or a result"
  | Unknown | Need_meet | Cannot_reify -> "unknown"
  | Invalid -> "invalid"
  | Stops _ -> "no answer"

let print_question ppf
    { about = { name; kind = _; mode };
      input_of;
      question =
        { asked;
          other;
          arguments =
            { min_name_mode;
              kind;
              field_index;
              function_slot;
              value_slot;
              variable;
              allowed_in = _;
              at_toplevel;
              symbol_projections
            };
          asked_of_compiler;
          other_of_compiler
        }
    } =
  let variables = Model_type.print_list Model_type.print_variable in
  Format.fprintf ppf
    "@[<v>of %a (which the environment has at the mode %s): %s@ the type: %a@ \
     as the compiler has it: %a@ the second type: %a@ as the compiler has it: \
     %a@ @[<hov 2>the arguments:@ min_name_mode %s,@ kind %a,@ field %d,@ \
     function slot %a,@ value slot %a,@ variable %a;@ of reification:@ at \
     toplevel %a (or none),@ projections from symbols %a@]@]"
    Name.print name (print_mode mode) (print_input_of input_of) Model_type.print
    asked TG.print
    (Lazy.force asked_of_compiler)
    Model_type.print other TG.print
    (Lazy.force other_of_compiler)
    (match min_name_mode with Normal -> "normal" | In_types -> "in types")
    Model_type.print_kind kind field_index N.print_positive function_slot
    N.print_positive value_slot Model_type.print_variable variable variables
    at_toplevel variables symbol_projections

(* The questions of an environment, asked. What is returned is the number of
   questions, and the cases, each with what to show of it: a case is a triple of
   answers that is compared and on which the three do not agree. An answer of
   the compiler that the model has no value for is counted as not compared, and
   is no case. *)
type not_an_agreement =
  { prover : string;
    asked : asked;
    answers : answers;
    is_a_candidate : bool
  }

(* For the control of the statuses, and for nothing else: the prover, if any,
   whose answer in the compiler's replay, or in the model, is replaced by
   another answer, so that a run has failures of the replay, or candidates,
   where it has none ([-spoil-the-replay], [-spoil-the-model]). A run in which
   something is spoiled says so first. *)
let spoiled_in_the_replay : string option ref = ref None

let spoiled_in_the_model : string option ref = ref None

let another_answer (answer : A.answer) : A.answer =
  match answer with
  | Invalid -> Unknown
  | Stops _ | Proved _ | Unknown | Known_result _ | Need_meet | Lift _
  | Simple _ | Cannot_reify ->
    Invalid

let say_what_is_spoiled () =
  List.iter
    (fun (spoiled, where) ->
      match !spoiled with
      | Some prover ->
        Format.printf
          "SPOILED, for the control of the statuses: the answers of %s %s@."
          prover where
      | None -> ())
    [ spoiled_in_the_model, "in the model";
      spoiled_in_the_replay, "of the compiler in its replay" ]

let ask (counts : counts) (of_unit : of_unit) (replayed : replayed)
    (model_env : Model_env.t) (replayed_env : TE.t) :
    int * not_an_agreement list =
  let questions = questions of_unit replayed model_env in
  let provers_for = provers_for of_unit replayed model_env in
  let found = ref [] in
  List.iter
    (fun (asked : asked) ->
      count counts ("questions: " ^ print_input_of asked.input_of);
      List.iter
        (fun (prover : PT.prover) ->
          let count what = count counts (key_of_prover prover.prover what) in
          let unless_spoiled spoiled answer =
            match !spoiled with
            | Some spoiled when String.equal spoiled prover.prover ->
              another_answer answer
            | Some _ | None -> answer
          in
          match
            { in_simplify =
                PT.answer_of_compiler prover replayed.env asked.question;
              in_replay =
                unless_spoiled spoiled_in_the_replay
                  (PT.answer_of_compiler prover replayed_env asked.question);
              in_model =
                unless_spoiled spoiled_in_the_model
                  (prover.of_model model_env asked.question)
            }
          with
          | exception Converter.Not_in_the_model what ->
            count
              ("not compared: an answer of the compiler is not in the model: "
             ^ general what)
          | answers -> (
            count (class_of_answer answers.in_simplify);
            (match answers.in_simplify with
            | Stops why -> count ("with no answer: " ^ general why)
            | Lift _ -> count "of the results, lifted"
            | Proved _ | Unknown | Known_result _ | Need_meet | Invalid
            | Simple _ | Cannot_reify ->
              ());
            match verdict answers with
            | The_three_agree -> count agree
            | The_compiler_differs_in_its_replay ->
              count "THE COMPILER ANSWERS OTHERWISE IN ITS REPLAY";
              found
                := { prover = prover.prover;
                     asked;
                     answers;
                     is_a_candidate = false
                   }
                   :: !found
            | The_model_differs ->
              count "DISAGREE";
              found
                := { prover = prover.prover;
                     asked;
                     answers;
                     is_a_candidate = true
                   }
                   :: !found))
        (provers_for asked))
    questions;
  List.length questions, List.rev !found

let print_not_an_agreement ppf
    { prover;
      asked;
      answers = { in_simplify; in_replay; in_model };
      is_a_candidate
    } =
  Format.fprintf ppf
    "@[<v 2>%s, %s:@ %a@ the compiler, in Simplify's environment: %a@ the \
     compiler, in its replay: %a@ the model: %a@]"
    (if is_a_candidate
     then
       "DISAGREEMENT (the two answers of the compiler agree, the model's is \
        another)"
     else "THE COMPILER ANSWERS OTHERWISE IN ITS REPLAY")
    prover print_question asked A.print in_simplify A.print in_replay A.print
    in_model

(* ======================================================================== *)
(* One unit                                                                 *)
(* ======================================================================== *)

let observations : (TE.t * Continuation.t) list ref = ref []

(* The observer only keeps what it is given: the environments are gone through
   when Simplify has returned. *)
let observe env continuation =
  observations := (env, continuation) :: !observations

(* The unit of a file, simplified, or why there is none. The text may be one
   that the parser or the conversion to a term refuses, and Simplify may stop on
   a unit, with a fatal error, at an assertion or with another exception: some
   of the units of the semantics tests are there to show it. *)
let simplify filename : (Flambda_unit.t, string) result =
  match Parse_flambda.parse_fexpr filename with
  | exception Failure what -> Error ("not read: the parser fails: " ^ what)
  | Error _ -> Error "not read: the parser refuses it"
  | Ok fexpr -> (
    let unit_info = Parse_flambda.make_unit_info ~filename in
    Env.set_current_unit unit_info;
    match
      Misc.Colours.without_colours ~f:(fun () ->
          or_fatal_error (fun () ->
              Fexpr_to_flambda.conv (Unit_info.modname unit_info) fexpr))
    with
    | exception Failure what ->
      Error ("not read: where the text is made a term: " ^ what)
    | Error line ->
      Error
        ("not read: a fatal error where the text is made a term: "
       ^ general line)
    | Ok converted -> (
      let null_formatter =
        Format.make_formatter (fun _ _ _ -> ()) (fun () -> ())
      in
      TE.For_testing.set_observer (Some observe);
      let result =
        Fun.protect
          ~finally:(fun () -> TE.For_testing.set_observer None)
          (fun () ->
            Misc.Colours.without_colours ~f:(fun () ->
                or_fatal_error (fun () ->
                    match
                      Flambda2.flambda_to_flambda ~ppf_dump:null_formatter
                        ~prefixname:(Filename.chop_extension filename)
                        ~machine_width
                        ~code_slot_offsets:converted.code_slot_offsets
                        converted.unit
                    with
                    | result -> Ok result.flambda
                    | exception Assert_failure (file, _, _) ->
                      Error
                        ("not simplified: an assertion of "
                       ^ Filename.basename file ^ " fails")
                    | exception
                        ((Invalid_argument _ | Failure _ | Not_found) as exn) ->
                      Error
                        ("not simplified: an exception: "
                       ^ Printexc.to_string exn))))
      in
      match result with
      | Ok result -> result
      | Error line -> Error ("not simplified: a fatal error: " ^ general line)))

let admitted = "admitted: the model's replay is seen as Simplify's environment"

let questions_asked = "questions, asked of the table and of reification twice"

(* The cases of a file, of each of the two kinds. A file may have both; each is
   counted, in the counts of the file. *)
type found =
  { candidates : int;
    failures_of_the_replay : int
  }

let units_with_a_disagreement = "units with a DISAGREEMENT"

let units_with_a_failure_of_the_replay =
  "units in which THE COMPILER ANSWERS OTHERWISE IN ITS REPLAY"

(* The status with which the run on one file is to end where its check returns:
   1 where a question is a candidate, else 4 where the compiler answers
   otherwise in its replay, else 0. A file that has both ends with 1. *)
let status_of (found : found) : int =
  if found.candidates > 0
  then 1
  else if found.failures_of_the_replay > 0
  then 4
  else 0

(* The names of an environment, in the order in which they are replayed
   ([names_of]: the symbols, then the variables in the order of binding): each
   with its number in the model, and what the two replays are made from, which
   is its kind, the mode at which the environment has it, its canonical element
   and its type, the last two as the model has them and as the compiler has
   them. *)
let print_names (numbering : Model_names.t) ppf (replayed : replayed) =
  Format.fprintf ppf "@[<v 2>the names, in the order of the replay:";
  List.iter
    (fun { of_simplify = { name_of_env = { name; kind; mode }; t; canonical };
           t_in_model;
           canonical_in_model
         } ->
      Format.fprintf ppf
        "@ @[<v 2>%a = %a, of kind %a, at the mode %s@ in the model, canonical \
         %a, with the type@ %a@ in the compiler, canonical %a, with the type@ \
         %a@]"
        Model_type.print_name
        (Converter.name_to_model numbering name)
        Name.print name K.print kind (print_mode mode) Model_type.print_simple
        canonical_in_model Model_type.print t_in_model Simple.print canonical
        TG.print t)
    replayed.read;
  Format.fprintf ppf "@]"

let with_the_replay =
  "admitted, and the compiler's replay is seen as Simplify's environment too"

(* One file: its environments are replayed and counted, and the questions are
   asked in those that are admitted with both replays. A case (a triple of
   answers that is compared and on which the three do not agree) is shown,
   whatever the options, and after the cases of an environment come the names of
   that environment. With [why], each environment that is not admitted with both
   replays is shown too, and with [names] its names. [flags] is what is said of
   the flags where a case is shown. *)
let check_file ~(why : bool) ~(names : bool) ~(flags : string) filename :
    counts * found =
  let shown = Filename.basename filename in
  let counts = create_counts () in
  let count = count counts in
  let candidates = ref 0 and failures_of_the_replay = ref 0 in
  observations := [];
  (match simplify filename with
  | Error why ->
    count "units not simplified";
    Format.printf "%s: %s@." shown why
  | Ok (_ : Flambda_unit.t) ->
    count "units simplified";
    let of_unit = of_unit () in
    let numbering = of_unit.numbering in
    let environments = List.rev !observations in
    List.iteri
      (fun i (env, (continuation : Continuation.t)) ->
        count "environments";
        let show (replayed : replayed option) what =
          if why
          then (
            Format.printf "@[<v 2>environment %d, at %a: %s" i
              Continuation.print continuation what;
            (match replayed with
            | None -> ()
            | Some replayed ->
              List.iter
                (fun (title, (compared : compared)) ->
                  match compared with
                  | Seen_the_same | Not_compared _ -> ()
                  | Differ { what; shown } ->
                    Format.fprintf Format.std_formatter "@ @[<v 2>%s: %s@ %t@]"
                      title what shown)
                [ ( "Simplify's against the model's replay",
                    replayed.simplify_against_model );
                  ( "Simplify's against the compiler's replay",
                    replayed.simplify_against_replay );
                  ( "the compiler's replay against the model's",
                    replayed.replay_against_model ) ]);
            (match replayed with
            | Some replayed when names ->
              Format.printf "@ %a" (print_names numbering) replayed
            | Some _ | None -> ());
            Format.printf "@]@.")
        in
        match replay numbering env with
        | Error why ->
          count ("lost: " ^ why);
          show None ("lost: " ^ why)
        | Ok replayed -> (
          let of_another_unit ~variable =
            List.exists
              (fun { name; _ } ->
                Bool.equal variable (Name.is_var name)
                && not (Current_unit.is_current (Name.compilation_unit name)))
              replayed.names
          in
          match replayed.simplify_against_model with
          | Seen_the_same -> (
            count admitted;
            add counts "names in the environments admitted"
              (List.length replayed.names);
            if of_another_unit ~variable:false
            then count "admitted, with a symbol of another unit in its names";
            if of_another_unit ~variable:true
            then count "admitted, with a variable of another unit in its names";
            match replayed.simplify_against_replay with
            | Seen_the_same -> (
              count with_the_replay;
              match replayed.model_env, replayed.replayed_env with
              | Ok model_env, Ok replayed_env ->
                let questions, found =
                  ask counts of_unit replayed model_env replayed_env
                in
                add counts questions_asked questions;
                List.iter
                  (fun (not_an_agreement : not_an_agreement) ->
                    incr
                      (if not_an_agreement.is_a_candidate
                       then candidates
                       else failures_of_the_replay);
                    Format.printf
                      "@[<v 2>%s, at %s, environment %d, at %a:@ %a@]@." shown
                      flags i Continuation.print continuation
                      print_not_an_agreement not_an_agreement)
                  found;
                if not (List.is_empty found)
                then
                  Format.printf
                    "@[<v 2>%s, at %s, environment %d, at %a, in which that \
                     was asked:@ %a@]@."
                    shown flags i Continuation.print continuation
                    (print_names numbering) replayed
              | Error _, _ | _, Error _ ->
                (* An environment that is seen the same was made. *)
                assert false)
            | (Differ _ | Not_compared _) as compared ->
              let what =
                "admitted, but the compiler's replay is not seen as Simplify's \
                 environment: " ^ what_of compared
              in
              count what;
              show (Some replayed) what)
          | (Differ _ | Not_compared _) as compared ->
            (* The model's replay is not Simplify's environment. Either the
               compiler's replay is not either, and then the replay does not
               give the environment back; or it is, and then the model's replay
               and the compiler's, made by the same steps, differ. *)
            let what =
              Printf.sprintf
                "lost: the model's replay is not seen as Simplify's \
                 environment (%s); the compiler's replay against Simplify's: \
                 %s; against the model's: %s"
                (what_of compared)
                (what_of replayed.simplify_against_replay)
                (what_of replayed.replay_against_model)
            in
            count what;
            show (Some replayed) what))
      environments;
    Format.printf
      "%s: %d environments, %d admitted, %d of them with the compiler's \
       replay; %d questions@."
      shown (List.length environments) (number counts admitted)
      (number counts with_the_replay)
      (number counts questions_asked));
  if !candidates > 0 then count units_with_a_disagreement;
  if !failures_of_the_replay > 0 then count units_with_a_failure_of_the_replay;
  ( counts,
    { candidates = !candidates;
      failures_of_the_replay = !failures_of_the_replay
    } )

(* The compiler keeps what it knows of a unit when it goes on to another one in
   the same run. So each file is gone through in a run of its own, from the
   directory of the file: this program runs itself on each, and adds up the
   counts that each run writes. The run on a file counts itself whether the file
   has a candidate, and whether it has a question on which the compiler answers
   otherwise in its replay: a file may have both, and its status says only the
   first. A run that stops is a line for its input, and is counted; so is a run
   whose status is not the one of what it counted, or that counted no unit. The
   tool ends with status 1 where a file has a candidate; else with 4 where one
   has a failure of the replay; else with 3 where a run stopped; else with 0.
   What is printed at the end says each of the three that there is. *)
let units_stopped_on = "units on which this tool stopped"

let each_in_a_run_of_its_own ~(flags : string list) (inputs : string list) =
  let totals = create_counts () in
  List.iter
    (fun input ->
      let counts_file =
        Filename.temp_file "provers_of_compilations" ".counts"
      in
      let status =
        Sys.command
          (Filename.quote_command Sys.executable_name
             (flags @ ["-one"; counts_file; input]))
      in
      let of_the_file = create_counts () in
      add_counts of_the_file counts_file;
      Sys.remove counts_file;
      let number = number of_the_file in
      let status_of_the_counts =
        status_of
          { candidates = number units_with_a_disagreement;
            failures_of_the_replay = number units_with_a_failure_of_the_replay
          }
      in
      if
        status = status_of_the_counts
        && number "units simplified" + number "units not simplified" = 1
      then
        List.iter
          (fun what -> add totals what (number what))
          (List.rev of_the_file.order)
      else (
        count totals units_stopped_on;
        Format.printf "%s: this tool stopped with status %d@."
          (Filename.basename input) status))
    inputs;
  Format.printf "@.In all:@.";
  print_counts totals;
  let disagreements = number totals units_with_a_disagreement
  and failures = number totals units_with_a_failure_of_the_replay
  and stopped = number totals units_stopped_on in
  Format.printf "@.";
  if disagreements > 0
  then Format.printf "DISAGREEMENTS in %d files.@." disagreements;
  if failures > 0
  then
    Format.printf "The compiler answers otherwise in its replay in %d files.@."
      failures;
  if disagreements = 0 && failures = 0
  then
    if stopped > 0
    then
      Format.printf
        "No disagreement in the files gone through; this tool stopped on %d.@."
        stopped
    else Format.printf "No disagreement.@."
  else if stopped > 0
  then Format.printf "This tool stopped on %d files.@." stopped;
  if disagreements > 0 then exit 1;
  if failures > 0 then exit 4;
  if stopped > 0 then exit 3

(* A control of the replay and of the gate, on an environment made by hand that
   has what the environments of the units of the tests may not have: variables
   bound at each of the three name modes, in scope and put out of scope by a
   closure ([TE.closure_env]), and a name in scope that is an alias of one out
   of scope. The mode at which the environment has a variable is to be the mode
   it was bound with where it is in scope, and the mode for types where it is
   not; and the environment is to be admitted, with the compiler's replay seen
   as it too. *)
let control () =
  Env.set_current_unit
    (Unit_info.make_dummy ~input_name:"provers_of_compilations"
       (Compilation_unit.of_string "Provers_of_compilations"));
  let variables = ref [] in
  let define env label (bound : mode) ~(expected : mode) =
    let variable = Variable.create label K.value in
    variables := (variable, label, bound, expected) :: !variables;
    ( TE.add_definition env
        (Bound_name.create_var
           (Bound_var.create variable Flambda_debug_uid.none
              (mode_of_compiler bound)))
        K.value,
      variable )
  in
  let env = TE.create ~machine_width ~resolver:(fun _ -> None) in
  let env, outside = define env "normal, outside" Normal ~expected:In_types in
  let env, _ = define env "phantom, outside" Phantom ~expected:In_types in
  let env, _ = define env "for types, outside" In_types ~expected:In_types in
  let env = TE.closure_env env in
  let env, inside = define env "normal, inside" Normal ~expected:Normal in
  let env, _ = define env "phantom, inside" Phantom ~expected:Phantom in
  let env, _ = define env "for types, inside" In_types ~expected:In_types in
  let env =
    match
      TE.add_alias env ~canonical_element1:(Simple.var outside)
        ~canonical_element2:(Simple.var inside)
    with
    | Ok { t = env; canonical_element = _; demoted_name = _ } ->
      TE.replace_equation env (Name.var inside)
        (TG.alias_type_of K.value (Simple.var outside))
    | Unknown | Bottom -> Misc.fatal_error "The control: no alias"
  in
  Format.printf
    "A control: an environment made by hand, with variables of the three name \
     modes, in the scope of a closure and outside it@.";
  let failed = ref false in
  let names = names_of env in
  List.iter
    (fun (variable, label, bound, expected) ->
      let found =
        List.find_map
          (fun { name; mode; kind = _ } ->
            if Name.equal name (Name.var variable) then Some mode else None)
          names
      in
      let as_expected =
        match found with
        | Some mode -> Stdlib.( = ) mode expected
        | None -> false
      in
      if not as_expected then failed := true;
      Format.printf "  %s, bound at the mode %s: the environment has it %s%s@."
        label (print_mode bound)
        (match found with
        | Some mode -> "at the mode " ^ print_mode mode
        | None -> "not")
        (if as_expected then "" else " (NOT AS EXPECTED)"))
    (List.rev !variables);
  (match replay (Model_names.create ()) env with
  | Error why ->
    failed := true;
    Format.printf "  the environment is LOST: %s@." why
  | Ok
      { simplify_against_model;
        simplify_against_replay;
        replay_against_model;
        _
      } ->
    List.iter
      (fun (title, (compared : compared)) ->
        (match compared with
        | Seen_the_same -> ()
        | Differ _ | Not_compared _ -> failed := true);
        Format.printf "  %s: %s@." title (what_of compared))
      [ "Simplify's against the model's replay", simplify_against_model;
        "Simplify's against the compiler's replay", simplify_against_replay;
        "the compiler's replay against the model's", replay_against_model ]);
  Format.printf "@.";
  if !failed then exit 1

(* A control of the statuses, on a file. This tool runs itself on the file, as
   it is run by hand and by the alias: with nothing spoiled; with the answers of
   a prover spoiled in the model, which makes candidates; with those of another
   spoiled in the compiler's replay, which makes failures of the replay; and
   with both, so that the one file has the two. Each time it is run on the file
   alone, in the three ways ([-one], [-why], [-why-with-names]); over the file
   as over a list; and over the file and one that is in no directory, on which
   the tool stops. What is printed: the status of each run; of the run on the
   file alone, the two counts of units, the counts of the cases of the two
   provers, and how many times the names of an environment are shown, which is
   to be once for each environment that has a case; and the last lines of the
   runs over a list. The statuses and the counts of units are to be those
   written here: the control fails if one is not. *)
let control_of_the_statuses ~(flags : string list) filename =
  let run ?stdout arguments =
    Sys.command
      (Filename.quote_command Sys.executable_name
         ~stdout:(Option.value stdout ~default:Filename.null)
         ~stderr:Filename.null arguments)
  in
  let lines path =
    let channel = open_in path in
    let rec read lines =
      match input_line channel with
      | line -> read (line :: lines)
      | exception End_of_file ->
        close_in channel;
        List.rev lines
    in
    read []
  in
  (* The lines after the last empty one. *)
  let last_lines path =
    List.fold_left
      (fun last line -> if String.equal line "" then [] else line :: last)
      [] (lines path)
    |> List.rev
  in
  let failed = ref false in
  let checked ~expected n =
    if n = expected
    then string_of_int n
    else (
      failed := true;
      Printf.sprintf "%d (NOT AS EXPECTED, which is %d)" n expected)
  in
  let in_the_model = "prove_equals_tagged_immediates"
  and in_the_replay = "meet_naked_immediates" in
  let the_model = ["-spoil-the-model"; in_the_model]
  and the_replay = ["-spoil-the-replay"; in_the_replay] in
  let in_no_directory = Filename.concat "no such directory" "unit.fl" in
  Format.printf
    "A control of the statuses: this tool, run by itself on %s with answers \
     spoiled@."
    (Filename.basename filename);
  List.iter
    (fun (title, spoiling, (candidates, failures), (expected, with_a_stop)) ->
      let flags = flags @ spoiling in
      let counts_file = Filename.temp_file "provers_of_compilations" ".counts"
      and output = Filename.temp_file "provers_of_compilations" ".output" in
      let one = run ~stdout:output (flags @ ["-one"; counts_file; filename]) in
      let counts = create_counts () in
      add_counts counts counts_file;
      let names_shown =
        let shown = "the names, in the order of the replay:" in
        List.length
          (List.filter
             (fun line -> String.ends_with ~suffix:shown (String.trim line))
             (lines output))
      in
      let why = run (flags @ ["-why"; filename]) in
      let why_with_names = run (flags @ ["-why-with-names"; filename]) in
      let disagree = key_of_prover in_the_model "DISAGREE"
      and otherwise =
        key_of_prover in_the_replay
          "THE COMPILER ANSWERS OTHERWISE IN ITS REPLAY"
      in
      Format.printf
        "  %s:@.    the file alone: status %s; with -why, %s; with \
         -why-with-names, %s@.    counted of it: %s %s; %s %s@.    of the \
         questions: %d DISAGREE; %d THE COMPILER ANSWERS OTHERWISE IN ITS \
         REPLAY@.    the names of an environment are shown %d times@."
        title (checked ~expected one) (checked ~expected why)
        (checked ~expected why_with_names)
        (checked ~expected:candidates (number counts units_with_a_disagreement))
        units_with_a_disagreement
        (checked ~expected:failures
           (number counts units_with_a_failure_of_the_replay))
        units_with_a_failure_of_the_replay (number counts disagree)
        (number counts otherwise) names_shown;
      List.iter
        (fun (what, files, expected) ->
          let over = run ~stdout:output (flags @ files) in
          Format.printf "    %s: status %s@." what (checked ~expected over);
          List.iter
            (fun line -> Format.printf "      %s@." line)
            (last_lines output))
        [ "over the file, as over a list", [filename], expected;
          ( "over the file and one that is in no directory",
            [filename; in_no_directory],
            with_a_stop ) ];
      Sys.remove counts_file;
      Sys.remove output)
    [ "nothing spoiled", [], (0, 0), (0, 3);
      "the answers of a prover spoiled in the model", the_model, (1, 0), (1, 1);
      ( "the answers of a prover spoiled in the compiler's replay",
        the_replay,
        (0, 1),
        (4, 4) );
      "both, of two provers", the_model @ the_replay, (1, 1), (1, 1) ];
  Format.printf "@.";
  if !failed then exit 1

let usage () =
  prerr_endline
    "usage: provers_of_compilations [-O3] FILE.fl ..., or \
     provers_of_compilations [-O3] -why FILE.fl, or provers_of_compilations \
     [-O3] -why-with-names FILE.fl, or provers_of_compilations -control, or \
     provers_of_compilations -control-of-the-statuses FILE.fl";
  exit 2

let () =
  (* The options that come first, which a run over files gives to the run on
     each. *)
  let rec options ~o3 flags arguments =
    match arguments with
    | ("-O3" as flag) :: arguments -> options ~o3:true (flag :: flags) arguments
    | ("-spoil-the-model" as flag) :: prover :: arguments ->
      spoiled_in_the_model := Some prover;
      options ~o3 (prover :: flag :: flags) arguments
    | ("-spoil-the-replay" as flag) :: prover :: arguments ->
      spoiled_in_the_replay := Some prover;
      options ~o3 (prover :: flag :: flags) arguments
    | arguments -> o3, List.rev flags, arguments
  in
  let o3, flags, arguments =
    options ~o3:false [] (List.tl (Array.to_list Sys.argv))
  in
  let one ~why ?(names = false) filename =
    say_what_is_spoiled ();
    Clflags.set_target Clflags.Target.Native;
    if o3
    then (
      Clflags.Opt_flag_handler.set Oxcaml_flags.opt_flag_handler;
      Clflags.set_o3 ());
    Sys.chdir (Filename.dirname filename);
    check_file ~why ~names
      ~flags:(if o3 then "-O3" else "the default flags")
      (Filename.basename filename)
  in
  let end_with (found : found) =
    match status_of found with 0 -> () | status -> exit status
  in
  match arguments with
  | ["-one"; counts_file; filename] ->
    let counts, found = one ~why:false filename in
    write_counts counts counts_file;
    end_with found
  | ["-why"; filename] ->
    let (_ : counts), found = one ~why:true filename in
    end_with found
  | ["-why-with-names"; filename] ->
    let (_ : counts), found = one ~why:true ~names:true filename in
    end_with found
  | ["-control"] ->
    Clflags.set_target Clflags.Target.Native;
    control ()
  | ["-control-of-the-statuses"; filename] ->
    control_of_the_statuses ~flags filename
  | []
  | ( "-one" | "-why" | "-why-with-names" | "-control"
    | "-control-of-the-statuses" )
    :: _ ->
    usage ()
  | filenames ->
    say_what_is_spoiled ();
    Format.printf
      "The environments that Simplify computes for the units of the semantics \
       tests%s, replayed into the Rocq model.@.@."
      (if o3 then ", at -O3" else "");
    each_in_a_run_of_its_own ~flags filenames
