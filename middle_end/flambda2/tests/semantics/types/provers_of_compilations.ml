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

(* The environments that Simplify computes, replayed into the Rocq model.

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

   What is printed for a file is a line: its environments, those that the
   model's replay is admitted for, and those of them for which the compiler's
   replay is seen as E too; and at the end, for all the files, the counts of the
   losses by reason. With [-why] a file is gone through alone and each
   environment that is not admitted with both replays is shown.

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

(* ======================================================================== *)
(* The names of an environment                                              *)
(* ======================================================================== *)

type mode =
  | Normal
  | In_types
  | Phantom

let modes = [Normal; In_types; Phantom]

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
   says that the name is there. For a variable that was bound outside the
   closure that Simplify is in, that is the mode for types, whatever the mode it
   was bound with: the environment gives it no other. *)
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

(* The names of an environment: the variables that its levels above the first
   define and the symbols that it defines, as in types_of_compilations.ml; and
   with them every name that the type of one of these mentions, and every name
   that the environment has as an alias of one, and so on. (The variables of the
   first level cannot be listed: [TE.cut] gives the levels above a scope, and
   there is none below the first.) A name that a type mentions and the
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

let print_counts counts =
  List.iter
    (fun what ->
      Format.printf "  %d %s@." (Hashtbl.find counts.table what) what)
    (List.rev counts.order)

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

let with_the_replay =
  "admitted, and the compiler's replay is seen as Simplify's environment too"

(* One file: its environments are replayed and counted. With [why], each
   environment that is not admitted with both replays is shown. *)
let check_file ~(why : bool) ~(names : bool) filename : counts =
  let shown = Filename.basename filename in
  let counts = create_counts () in
  let count = count counts in
  observations := [];
  (match simplify filename with
  | Error why ->
    count "units not simplified";
    Format.printf "%s: %s@." shown why
  | Ok (_ : Flambda_unit.t) ->
    count "units simplified";
    let numbering = Model_names.create () in
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
              Format.printf "@ @[<v 2>the names, in the order of the replay:";
              List.iter
                (fun { of_simplify = { name_of_env = { name; mode; _ }; _ };
                       t_in_model;
                       canonical_in_model
                     } ->
                  Format.printf "@ %a = %a, %s, canonical %a: %a"
                    Model_type.print_name
                    (Converter.name_to_model numbering name)
                    Name.print name (print_mode mode) Model_type.print_simple
                    canonical_in_model Model_type.print t_in_model)
                replayed.read;
              Format.printf "@]"
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
            | Seen_the_same -> count with_the_replay
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
       replay@."
      shown (List.length environments) (number counts admitted)
      (number counts with_the_replay));
  counts

(* The compiler keeps what it knows of a unit when it goes on to another one in
   the same run. So each file is gone through in a run of its own, from the
   directory of the file: this program runs itself on each, and adds up the
   counts that each run writes. A run that stops is a line for its input, and is
   counted: the tool then ends with status 3. *)
let each_in_a_run_of_its_own ~(flags : string list) (inputs : string list) =
  let totals = create_counts () in
  List.iter
    (fun input ->
      let counts_file =
        Filename.temp_file "provers_of_compilations" ".counts"
      in
      match
        Sys.command
          (Filename.quote_command Sys.executable_name
             (flags @ ["-one"; counts_file; input]))
      with
      | 0 ->
        add_counts totals counts_file;
        Sys.remove counts_file
      | status ->
        count totals "units on which this tool stopped";
        Format.printf "%s: this tool stopped with status %d@."
          (Filename.basename input) status)
    inputs;
  Format.printf "@.In all:@.";
  print_counts totals;
  let stopped = number totals "units on which this tool stopped" in
  if stopped > 0 then exit 3

let usage () =
  prerr_endline
    "usage: provers_of_compilations [-O3] FILE.fl ..., or \
     provers_of_compilations [-O3] -why FILE.fl";
  exit 2

let () =
  let o3, arguments =
    match List.tl (Array.to_list Sys.argv) with
    | "-O3" :: arguments -> true, arguments
    | arguments -> false, arguments
  in
  let one ~why ?(names = false) filename =
    Clflags.set_target Clflags.Target.Native;
    if o3
    then (
      Clflags.Opt_flag_handler.set Oxcaml_flags.opt_flag_handler;
      Clflags.set_o3 ());
    Sys.chdir (Filename.dirname filename);
    check_file ~why ~names (Filename.basename filename)
  in
  match arguments with
  | ["-one"; counts_file; filename] ->
    write_counts (one ~why:false filename) counts_file
  | ["-why"; filename] -> ignore (one ~why:true filename : counts)
  | ["-why-with-names"; filename] ->
    ignore (one ~why:true ~names:true filename : counts)
  | [] | ("-one" | "-why" | "-why-with-names") :: _ -> usage ()
  | filenames ->
    Format.printf
      "The environments that Simplify computes for the units of the semantics \
       tests%s, replayed into the Rocq model.@.@."
      (if o3 then ", at -O3" else "");
    each_in_a_run_of_its_own ~flags:(if o3 then ["-O3"] else []) filenames
