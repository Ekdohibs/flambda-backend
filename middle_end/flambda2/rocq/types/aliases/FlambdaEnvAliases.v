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

(* The aliases of the typing environment: which names and constants are
   known to be equal, and the canonical element of each class.

   This file is the model's copy of
   middle_end/flambda2/types/env/aliases.ml: the structure and the
   functions on it, written as the OCaml is, function by function and
   under its names, in a module [Aliases]. It extracts. What the
   structure means and what the functions guarantee is
   FlambdaEnvAliasesSpec.v, whose header says what is to be read here
   against the OCaml and lists every difference; in short:
   - a [Simple] has no coercion (FlambdaModel.v), so the coercions that
     the OCaml keeps beside each alias are gone, with their inverses
     and compositions;
   - the model has one compilation unit;
   - [Aliases_of_canonical_element.t] is one map, from each alias to its
     name mode, where the OCaml has two fields that say the same;
   - a fatal error of the OCaml that no flag guards is [None]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2 Require FlambdaCompareSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Names and simples                                                        *)
(* ======================================================================== *)

(* [Name.is_symbol]. *)
Definition name_is_symbol (n : name) : bool :=
  match n with
  | Name_symbol _ => true
  | Name_var _ => false
  end.

(* What [Simple.pattern_match] tells: a simple is a constant or a name.
   This is the name, if it is one. *)
Definition name_of_simple (s : simple) : option name :=
  match s with
  | Var x => Some (Name_var x)
  | Symbol sym => Some (Name_symbol sym)
  | Const _ => None
  end.

(* [Reg_width_const.equal]: the equality test of constants, the one of
   FlambdaCompareSpec.v. *)
Abbreviation const_eqb := FlambdaCompareSpec.const_eqb.

(* [Simple.equal]. *)
Definition simple_equal (s s' : simple) : bool :=
  match s, s' with
  | Var x, Var x' => Pos.eqb x x'
  | Symbol sym, Symbol sym' => Pos.eqb sym sym'
  | Const c, Const c' => const_eqb c c'
  | _, _ => false
  end.

Module Aliases.

(* ======================================================================== *)
(* The aliases of one canonical element                                     *)
(* ======================================================================== *)

Module Aliases_of_canonical_element.

  (* The names that have been demoted to one canonical element, each
     with the name mode it had when it was demoted. In the OCaml this is
     a record of [aliases], the names grouped by name mode, and [all],
     the names; they are the names bound here to one mode ([at_mode])
     and the names bound here ([all]). *)
  Definition t : Type := Keyed.finite_map name Name_mode.t.

  Definition empty : t := [].

  Definition is_empty (t : t) : bool := Keyed.is_empty t.

  Definition mem (t : t) (elt : name) : bool := Keyed.is_bound name_eqb elt t.

  (* A name that is there already is a fatal error. *)
  Definition add (t : t) (elt : name) (name_mode : Name_mode.t)
      : option Aliases_of_canonical_element.t :=
    if mem t elt then None else Some ((elt, name_mode) :: t).

  Definition all (t : t) : list name := Keyed.domain name_eqb t.

  (* [Name_mode.Map.find order t.aliases], or the empty set. *)
  Definition at_mode (t : t) (order : Name_mode.t) : list name :=
    map fst
      (Keyed.filter_bindings name_eqb
         (fun _ name_mode => Name_mode.equal name_mode order) t).

  (* The aliases at the least name mode that is at least [min_name_mode]
     and has some, once [filter_by_scope] has been applied. The OCaml
     folds over a [Name_mode.Map.t], which goes through the name modes
     from the least to the greatest: [In_types], [Phantom], [Normal]. *)
  Definition find_earliest_candidates (t : t)
      (filter_by_scope : Name_mode.t -> list name -> list name)
      (min_name_mode : Name_mode.t) : option (list name) :=
    fold_left
      (fun res_opt order =>
         match res_opt with
         | Some _ => res_opt
         | None =>
           match Name_mode.compare_partial_order order min_name_mode with
           | None => None
           | Some result =>
             if 0 <=? result
             then
               let aliases := filter_by_scope order (at_mode t order) in
               if Keyed.is_empty aliases then None else Some aliases
             else None
           end
         end)
      [Name_mode.In_types; Name_mode.Phantom; Name_mode.Normal] None.

  (* The two are taken to have no name in common. *)
  Definition union (t1 t2 : t) : Aliases_of_canonical_element.t := t1 ++ t2.

End Aliases_of_canonical_element.

(* ======================================================================== *)
(* Sets of aliases                                                          *)
(* ======================================================================== *)

Module Alias_set.

  (* [Aliases.Alias_set.t]: the aliases of one simple, a constant at most
     and names. The fields [const] and [names] of the OCaml are
     [set_const] and [set_names]. *)
  Record t : Type := make {
    set_const : option const;
    set_names : list name
  }.

  Definition empty : t := make None [].

  Definition is_empty (t : t) : bool :=
    match set_const t with
    | Some _ => false
    | None => Keyed.is_empty (set_names t)
    end.

  Definition create_aliases_of_element (canonical_element : simple)
      (alias_names : list name) : t :=
    match canonical_element with
    | Const canonical_const => make (Some canonical_const) alias_names
    | Var var => make None (Name_var var :: alias_names)
    | Symbol symbol => make None (Name_symbol symbol :: alias_names)
    end.

  Definition singleton (simple : simple) : t :=
    match simple with
    | Const const => make (Some const) []
    | Var var => make None [Name_var var]
    | Symbol symbol => make None [Name_symbol symbol]
    end.

  (* Of several names the OCaml answers the one that an order on names
     puts first; the model answers the first of its list. A set that
     has a constant and names has no answer, in the OCaml as here. *)
  Definition choose_opt (t : t) : option simple :=
    match set_const t with
    | Some const =>
      if Keyed.is_empty (set_names t) then Some (Const const) else None
    | None =>
      match set_names t with
      | [] => None
      | name :: _ => Some (simple_of_name name)
      end
    end.

  Definition get_singleton (t : t) : option simple :=
    match set_const t with
    | Some const =>
      if Keyed.is_empty (set_names t) then Some (Const const) else None
    | None =>
      option_map simple_of_name (Keyed.get_singleton name_eqb (set_names t))
    end.

  Definition inter (t1 t2 : t) : t :=
    let const :=
      match set_const t1, set_const t2 with
      | Some const1, Some const2 =>
        if const_eqb const1 const2 then Some const1 else None
      | _, _ => None
      end
    in
    make const (Keyed.inter name_eqb (set_names t1) (set_names t2)).

  Definition filter (t : t) (f : simple -> bool) : Alias_set.t :=
    let const :=
      match set_const t with
      | Some const => if f (Const const) then Some const else None
      | None => None
      end
    in
    make const
      (List.filter (fun name => f (simple_of_name name)) (set_names t)).

  (* The best alias of the set: its constant; failing that a symbol;
     failing that a variable. Of several symbols, or of several
     variables, the OCaml answers the one that an order on names puts
     first; the model answers the first of its list. *)
  Definition find_best (t : t) : option simple :=
    match set_const t with
    | Some const => Some (Const const)
    | None =>
      let symbols := List.filter name_is_symbol (set_names t) in
      let vars :=
        List.filter (fun name => negb (name_is_symbol name)) (set_names t)
      in
      match symbols with
      | symbol :: _ => Some (simple_of_name symbol)
      | [] =>
        match vars with
        | var :: _ => Some (simple_of_name var)
        | [] => None
        end
      end
    end.

End Alias_set.

(* ======================================================================== *)
(* The structure                                                            *)
(* ======================================================================== *)

(* [Aliases.t]. [canonical_elements] gives each name that has been
   demoted the canonical element of its class; a canonical element is
   not bound there. The two other maps are its inverse: the names
   demoted to each canonical element, be it a name or a constant; a
   canonical element with no alias need not be bound there. *)
Record t : Type := make {
  canonical_elements : Keyed.finite_map name simple;
  aliases_of_canonical_names
    : Keyed.finite_map name Aliases_of_canonical_element.t;
  aliases_of_consts : Keyed.finite_map const Aliases_of_canonical_element.t
}.

Section Binding_times.

(* What the functions below take of the typing environment: for the names
   it has defined, when each was bound and its name mode
   ([binding_times_and_modes], which is the environment's map from names
   to their types, of which the types are not read), and the same for a
   name of another compilation unit ([binding_time_resolver]). Each
   function takes, in this order, those of the two that it uses. *)
Context {A : Type}
  (binding_time_resolver : name -> Binding_time.With_name_mode.t)
  (binding_times_and_modes
    : Keyed.finite_map name (A * Binding_time.With_name_mode.t)).

(* The order on names that chooses the canonical element: [alias] is
   earlier than [than]. *)
Definition name_defined_earlier (alias than : name) : bool :=
  if name_eqb alias than || (name_is_symbol alias && name_is_symbol than)
  then false
  else if name_is_symbol alias && negb (name_is_symbol than)
  then true
  else if name_is_symbol than && negb (name_is_symbol alias)
  then false
  else
    let alias_binding_time :=
      match Keyed.lookup name_eqb alias binding_times_and_modes with
      | None => Binding_time.imported_variables
      | Some (_, binding_time_and_mode) =>
        Binding_time.With_name_mode.binding_time binding_time_and_mode
      end
    in
    let than_binding_time :=
      match Keyed.lookup name_eqb than binding_times_and_modes with
      | None => Binding_time.imported_variables
      | Some (_, binding_time_and_mode) =>
        Binding_time.With_name_mode.binding_time binding_time_and_mode
      end
    in
    if Binding_time.strictly_earlier alias_binding_time than_binding_time
    then true
    else if negb (Binding_time.equal alias_binding_time than_binding_time)
    then false
    else
      (* The OCaml compares the compilation units of the two names here.
         The model has one. *)
      let alias_binding_time :=
        Binding_time.With_name_mode.binding_time (binding_time_resolver alias)
      in
      let than_binding_time :=
        Binding_time.With_name_mode.binding_time (binding_time_resolver than)
      in
      Binding_time.strictly_earlier alias_binding_time than_binding_time.

(* The same on simples: a constant is earlier than any name, and nothing
   is earlier than a constant. *)
Definition defined_earlier (alias than : simple) : bool :=
  match name_of_simple than with
  | None => false
  | Some than =>
    match name_of_simple alias with
    | None => true
    | Some alias => name_defined_earlier alias than
    end
  end.

Definition binding_time_and_name_mode (elt : simple)
    : Binding_time.With_name_mode.t :=
  match elt with
  | Const _ => Binding_time.With_name_mode.consts
  | Var var =>
    match Keyed.lookup name_eqb (Name_var var) binding_times_and_modes with
    | Some (_, binding_time_and_mode) => binding_time_and_mode
    | None => Binding_time.With_name_mode.imported_variables
    end
  | Symbol _ => Binding_time.With_name_mode.symbols
  end.

Definition compute_name_mode_unscoped (elt : simple) : Name_mode.t :=
  Binding_time.With_name_mode.name_mode (binding_time_and_name_mode elt).

Definition compute_name_mode (elt : simple)
    (min_binding_time : Binding_time.t) : Name_mode.t :=
  Binding_time.With_name_mode.scoped_name_mode
    (binding_time_and_name_mode elt) min_binding_time.

(* No name is in two of the sets. *)
Fixpoint disjoint_alias_sets (sets : list (list name)) : bool :=
  match sets with
  | [] => true
  | names :: sets =>
    forallb
      (fun names' => Keyed.is_empty (Keyed.inter name_eqb names names')) sets
    && disjoint_alias_sets sets
  end.

(* What the OCaml's [invariant] checks, when the flag for it is set: a
   canonical name is earlier than each of its aliases and is not one of
   them, and no name is an alias of two canonical elements. The answer
   is [true] where the OCaml goes on and [false] where it stops with a
   fatal error. ([Aliases_of_canonical_element.invariant], that its two
   fields agree, has nothing to check here.) *)
Definition invariant (t : t) : bool :=
  let of_names := Keyed.compact name_eqb (aliases_of_canonical_names t) in
  let of_consts := Keyed.compact const_eqb (aliases_of_consts t) in
  forallb
    (fun '(canonical_element, aliases) =>
       forallb (name_defined_earlier canonical_element)
         (Aliases_of_canonical_element.all aliases)
       && negb (Aliases_of_canonical_element.mem aliases canonical_element))
    of_names
  && disjoint_alias_sets
       (map (fun binding => Aliases_of_canonical_element.all (snd binding))
          of_names
        ++ map (fun binding => Aliases_of_canonical_element.all (snd binding))
             of_consts).

Definition empty : t := make [] [] [].

Definition is_empty (t : t) : bool :=
  Keyed.is_empty (canonical_elements t)
  && Keyed.is_empty (aliases_of_canonical_names t)
  && Keyed.is_empty (aliases_of_consts t).

(* The OCaml's type [canonical]. *)
Inductive is_canonical_or_alias : Type :=
  | Is_canonical
  | Alias_of_canonical (canonical_element : simple).

Definition canonical (t : t) (element : simple) : is_canonical_or_alias :=
  match name_of_simple element with
  | None => Is_canonical
  | Some name =>
    match Keyed.lookup name_eqb name (canonical_elements t) with
    | None => Is_canonical
    | Some canonical_element => Alias_of_canonical canonical_element
    end
  end.

Definition get_aliases_of_canonical_element (t : t)
    (canonical_element : simple) : Aliases_of_canonical_element.t :=
  let aliases :=
    match canonical_element with
    | Var var =>
      Keyed.lookup name_eqb (Name_var var) (aliases_of_canonical_names t)
    | Symbol symbol =>
      Keyed.lookup name_eqb (Name_symbol symbol)
        (aliases_of_canonical_names t)
    | Const const => Keyed.lookup const_eqb const (aliases_of_consts t)
    end
  in
  match aliases with
  | None => Aliases_of_canonical_element.empty
  | Some aliases => aliases
  end.

(* [to_be_demoted] and its aliases become aliases of [canonical_element].
   The two are canonical. [None] is the fatal error of
   [Aliases_of_canonical_element.add]; [Bottom] is two constants. *)
Definition add_alias_between_canonical_elements (t : t)
    (canonical_element to_be_demoted : simple)
    : option (Or_bottom.t Aliases.t) :=
  if simple_equal canonical_element to_be_demoted
  then Some (Or_bottom.Ok t)
  else
    match name_of_simple to_be_demoted with
    | None => Some Or_bottom.Bottom
    | Some name_to_be_demoted =>
      let aliases_of_to_be_demoted :=
        get_aliases_of_canonical_element t to_be_demoted
      in
      let canonical_elements :=
        (name_to_be_demoted, canonical_element)
        :: map (fun alias => (alias, canonical_element))
             (Aliases_of_canonical_element.all aliases_of_to_be_demoted)
        ++ canonical_elements t
      in
      let aliases_of_canonical_element :=
        get_aliases_of_canonical_element t canonical_element
      in
      let* aliases :=
        Aliases_of_canonical_element.add
          (Aliases_of_canonical_element.union aliases_of_to_be_demoted
             aliases_of_canonical_element)
          name_to_be_demoted
          (compute_name_mode_unscoped to_be_demoted)
      in
      let aliases_of_canonical_names :=
        Keyed.unbind name_eqb name_to_be_demoted (aliases_of_canonical_names t)
      in
      let '(aliases_of_canonical_names, aliases_of_consts) :=
        match canonical_element with
        | Var var =>
          ((Name_var var, aliases) :: aliases_of_canonical_names,
           aliases_of_consts t)
        | Symbol symbol =>
          ((Name_symbol symbol, aliases) :: aliases_of_canonical_names,
           aliases_of_consts t)
        | Const const =>
          (aliases_of_canonical_names,
           (const, aliases) :: aliases_of_consts t)
        end
      in
      Some
        (Or_bottom.Ok
           (make canonical_elements aliases_of_canonical_names
              aliases_of_consts))
    end.

Inductive to_be_demoted : Type :=
  | Demote_canonical_element1
  | Demote_canonical_element2.

Definition choose_canonical_element_to_be_demoted
    (canonical_element1 canonical_element2 : simple) : to_be_demoted :=
  if defined_earlier canonical_element1 canonical_element2
  then Demote_canonical_element2
  else Demote_canonical_element1.

Definition add_alias (t : t) (canonical_element1 canonical_element2 : simple)
    : option (Or_bottom.t (Aliases.t * to_be_demoted)) :=
  let which_element :=
    choose_canonical_element_to_be_demoted canonical_element1
      canonical_element2
  in
  let '(canonical_element, demoted_canonical) :=
    match which_element with
    | Demote_canonical_element1 => (canonical_element2, canonical_element1)
    | Demote_canonical_element2 => (canonical_element1, canonical_element2)
    end
  in
  let* t :=
    add_alias_between_canonical_elements t canonical_element demoted_canonical
  in
  match t with
  | Or_bottom.Bottom => Some Or_bottom.Bottom
  | Or_bottom.Ok t => Some (Or_bottom.Ok (t, which_element))
  end.

(* In the OCaml [add0] takes the coercions off its arguments and gives
   them to [add_alias]. *)
Definition add0 (t : t) (canonical_element1 canonical_element2 : simple)
    : option (Or_bottom.t (Aliases.t * to_be_demoted)) :=
  add_alias t canonical_element1 canonical_element2.

(* [Aliases.add_result]. The field [t] of the OCaml is [add_result_t]. *)
Record add_result : Type := make_add_result {
  add_result_t : Aliases.t;
  demoted_name : name;
  canonical_element : simple
}.

(* What the OCaml's [invariant_add_result] checks of the result of [add],
   when the flag for it is set: [invariant], that the canonical element
   is earlier than the name demoted, and that this name is no longer
   canonical. *)
Definition invariant_add_result (result : add_result) : bool :=
  invariant (add_result_t result)
  && defined_earlier (canonical_element result)
       (simple_of_name (demoted_name result))
  && match
       canonical (add_result_t result) (simple_of_name (demoted_name result))
     with
     | Is_canonical => false
     | Alias_of_canonical _ => true
     end.

(* [Aliases.add]: the two, which are canonical and different, are now
   aliases. [None] is a fatal error; [Bottom] is two constants. *)
Definition add (t : t) (canonical_element1 canonical_element2 : simple)
    : option (Or_bottom.t add_result) :=
  let* result := add0 t canonical_element1 canonical_element2 in
  match result with
  | Or_bottom.Bottom => Some Or_bottom.Bottom
  | Or_bottom.Ok (t, which_element) =>
    let '(canonical_element, alias_of_demoted_element) :=
      match which_element with
      | Demote_canonical_element1 => (canonical_element2, canonical_element1)
      | Demote_canonical_element2 => (canonical_element1, canonical_element2)
      end
    in
    (* A constant cannot be demoted: a fatal error. *)
    let* name := name_of_simple alias_of_demoted_element in
    Some (Or_bottom.Ok (make_add_result t name canonical_element))
  end.

(* The earliest of the aliases of [canonical_element] that may be used at
   [min_name_mode] where the variables bound at [min_binding_time] or
   later are in scope. [None] is [Not_found]. Of several that are
   earliest, the OCaml takes one by an order on names that the model has
   not; the model takes the first of its list. *)
Definition find_earliest_alias (t : t) (canonical_element : simple)
    (min_binding_time : Binding_time.t) (min_name_mode : Name_mode.t)
    : option simple :=
  let aliases := get_aliases_of_canonical_element t canonical_element in
  let filter_by_scope name_mode names :=
    if Name_mode.equal name_mode Name_mode.in_types
    then names
    else
      filter
        (fun name =>
           let scoped_name_mode :=
             compute_name_mode (simple_of_name name) min_binding_time
           in
           Name_mode.equal name_mode scoped_name_mode)
        names
  in
  match
    Aliases_of_canonical_element.find_earliest_candidates aliases
      filter_by_scope min_name_mode
  with
  | Some at_earliest_mode =>
    match at_earliest_mode with
    | [] => None
    | first :: _ =>
      let earliest :=
        fold_left
          (fun min_elt elt =>
             if name_defined_earlier elt min_elt then elt else min_elt)
          at_earliest_mode first
      in
      Some (simple_of_name earliest)
    end
  | None => None
  end.

(* [Aliases.get_canonical_element_exn]: the canonical element of
   [element], whose name mode is [elt_name_mode], if it may be used at
   [min_name_mode]; failing that, the earliest alias that may be. [None]
   is [Not_found]. *)
Definition get_canonical_element_exn (t : t) (element : simple)
    (elt_name_mode min_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t) : option simple :=
  let at_least_min_name_mode name_mode :=
    match Name_mode.compare_partial_order name_mode min_name_mode with
    | None => false
    | Some c => 0 <=? c
    end
  in
  let canonical := canonical t element in
  match canonical, at_least_min_name_mode elt_name_mode with
  | Is_canonical, true => Some element
  | _, _ =>
    let '(canonical_element, name_mode) :=
      match canonical with
      | Is_canonical => (element, elt_name_mode)
      | Alias_of_canonical canonical_element =>
        (canonical_element,
         compute_name_mode canonical_element min_binding_time)
      end
    in
    if at_least_min_name_mode name_mode
    then Some canonical_element
    else find_earliest_alias t canonical_element min_binding_time min_name_mode
  end.

End Binding_times.

(* [Aliases.get_aliases]: the canonical element of [element] and its
   aliases, [element] among them. *)
Definition get_aliases (t : t) (element : simple) : Alias_set.t :=
  let canonical_element :=
    match canonical t element with
    | Is_canonical => element
    | Alias_of_canonical canonical_element => canonical_element
    end
  in
  Alias_set.create_aliases_of_element canonical_element
    (Aliases_of_canonical_element.all
       (get_aliases_of_canonical_element t canonical_element)).

Definition get_canonical_ignoring_name_mode (t : t) (name : name) : simple :=
  let elt := simple_of_name name in
  match canonical t elt with
  | Is_canonical => elt
  | Alias_of_canonical canonical_element => canonical_element
  end.

(* The function [add_alias] that the OCaml's [add_alias_set] defines for
   itself: [elt], which is canonical, becomes an alias of
   [canonical_element1], the canonical element so far, and the answer
   has the canonical element of the two. [None] is a fatal error: one of
   [add0], or two different constants ("Discovered Bottom during alias
   join"). *)
Definition add_alias_of_set {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (acc : t * simple) (elt : simple) : option (Aliases.t * simple) :=
  let '(t, canonical_element1) := acc in
  if simple_equal canonical_element1 elt
  then Some (t, canonical_element1)
  else
    match
      add0 binding_time_resolver binding_times_and_modes t canonical_element1
        elt
    with
    | None => None
    | Some Or_bottom.Bottom => None
    | Some (Or_bottom.Ok (t, which_element)) =>
      let canonical_element :=
        match which_element with
        | Demote_canonical_element1 => elt
        | Demote_canonical_element2 => canonical_element1
        end
      in
      Some (t, canonical_element)
    end.

(* [Aliases.add_alias_set]: [name] and every element of the set are now
   aliases. The OCaml goes through the names of the set in the order of
   names; the model in the order of its list. *)
Definition add_alias_set {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : t) (name : name) (aliases : Alias_set.t) : option Aliases.t :=
  let canonical := get_canonical_ignoring_name_mode t name in
  let* acc :=
    match Alias_set.set_const aliases with
    | None => Some (t, canonical)
    | Some c =>
      add_alias_of_set binding_time_resolver binding_times_and_modes
        (t, canonical) (Const c)
    end
  in
  let* acc :=
    fold_left
      (fun acc name =>
         let* acc := acc in
         let elt := get_canonical_ignoring_name_mode (fst acc) name in
         add_alias_of_set binding_time_resolver binding_times_and_modes acc
           elt)
      (Alias_set.set_names aliases) (Some acc)
  in
  Some (fst acc).

End Aliases.
