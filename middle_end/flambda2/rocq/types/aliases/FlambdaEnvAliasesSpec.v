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

(* The aliases of the typing environment: what the structure means, the
   order that chooses the canonical element of a class, and what the
   queries answer.

   This file is to be reviewed, with FlambdaEnvAliases.v, against
   middle_end/flambda2/types/env/aliases.ml and aliases.mli. In
   FlambdaEnvAliases.v, read against the OCaml: the type [Aliases.t], the
   order ([Aliases.name_defined_earlier], [Aliases.defined_earlier]) and
   the three functions that give a name its binding time and name mode
   ([Aliases.binding_time_and_name_mode], [Aliases.compute_name_mode],
   [Aliases.compute_name_mode_unscoped]). The queries and [Aliases.add]
   are written there as the OCaml is, but need not be read: what they
   answer is stated here. Binding times and name modes are
   FlambdaBindingTime.v.

   The structure. The simplifier learns that a name is equal to another
   name or to a constant (an [Equals] type), and keeps what it has
   learnt as classes of simples that are known to be equal, each with a
   canonical element. [Aliases.t] holds, for each name that is not the
   canonical element of its class (a name that has been demoted), its
   canonical element ([canonical_elements]), and the same the other way
   round, for each canonical element the names demoted to it
   ([aliases_of_canonical_names] and [aliases_of_consts]). A simple that
   the structure does not speak of is alone in its class.

   What it means ([aliases_respected]). The structure is a claim about
   the values of names when the program runs: an environment of the
   machine respects it when every name that has been demoted has the
   value of its canonical element.
   - The value of a simple is what [eval_simple] gives: a constant is
     its own value; a symbol is a pointer, a value by itself, whether
     or not the static environment binds it; a variable has the value
     the environment binds it to, and none if it binds it to nothing.
     So the static environment is not needed.
   - Values are compared as [eval_simple] gives them, a value or none.
     A variable aliased to a constant or to a symbol is therefore bound
     to it; two variables that are aliased are bound to the same value,
     or neither is bound. That every name the typing environment knows
     has a value is not said here: the names that exist only in types,
     or are out of scope, have none at run time, and the typing
     environment says what stands for their values
     (FlambdaTypeMeaningSpec.v, "The environment").
   - The values are those of the machine, and equal means the same
     value: a variable that holds a closure made by a [Let] is not
     equal to a symbol bound to a closure of the same code and slots.
   - Respect asks the same of the two inverse maps: a name listed among
     the aliases of a canonical element has its value. The three maps
     agree when the structure is consistent ([aliases_consistent]), and
     then this adds nothing; it is asked so that what a query answers
     has the value of what it was asked about with no condition on the
     structure ([respected_canonical_at_mode_statement]).

   The order ([Aliases.defined_earlier]). The canonical element of a
   class is its earliest element: a constant before any name; a symbol
   before any variable; of two variables the one bound first, by the
   binding times of the typing environment. The statements of the
   section "The order" say this of the function. Two symbols are not
   ordered, nor are two variables with the same binding times.

   What is stated of the queries:
   - [Aliases.get_canonical_ignoring_name_mode] answers the canonical
     element of the class ([canonical_element_of]), which is not itself
     demoted, is earlier than every other element of the class, is the
     constant of the class if it has one, and is not a variable if the
     class has a symbol; so two structures that have the invariant
     and the same classes have the same canonical elements
     ([canonical_element_determined_statement]);
   - [Aliases.get_canonical_element_exn] answers the canonical element
     if it may be used at the name mode asked for, and failing that an
     alias that may; what it answers is in the class of what it was
     asked about, and may be used at that name mode. Which alias: one
     of the least name mode that will do, and of those one that no other
     is earlier than ([earliest_alias_statement]); and there is no
     answer only if no alias may be used ([no_alias_statement]).

   Differences from the OCaml, all deliberate.
   - Coercions. A [Simple] of the model has no coercion
     (FlambdaModel.v), so the model's classes are of simples that are
     equal, where the OCaml's are of simples equal up to a coercion,
     and keep beside each alias the coercion to its canonical element.
     A coercion changes only the depth of recursion recorded for a
     closure ([Coercion.t] is [Id] or [Change_depth]), which the machine
     does not have: a simple and its coercion have the same value. So
     respect loses nothing. What is not modelled: that the coercions
     are composed and inverted correctly; the fatal errors of
     [Coercion.compose_exn] and of adding an alias of an element to
     itself with a coercion that is not the identity.
   - One compilation unit. [Aliases.name_defined_earlier] leaves out
     the comparison of the compilation units of two variables that have
     the same binding time, and goes to the last step, the binding
     times that [binding_time_resolver] gives. That function is total
     here. In the compiler it may raise ([Binding_time_resolver_failure],
     which [Typing_env] catches, dropping the alias), and for a name of
     the current unit it stops with a fatal error.
   - [Aliases_of_canonical_element.t] is one map from each alias to the
     name mode it had when it was demoted. The OCaml's record has
     [aliases], these names grouped by name mode, and [all], the names;
     its invariant that the two agree has nothing to say here.
   - The three maps of [Aliases.t] are kept as the OCaml has them, and
     not replaced by the one that determines the others, so that the
     functions read as the OCaml's do. That they agree is the condition
     [aliases_consistent], which is not checked by the OCaml's
     [invariant].
   - Maps are association lists of which the first binding of a key
     counts (FlambdaMapSpec.v): where the OCaml replaces a binding, the
     model adds one in front.
   - Fatal errors. Two of the OCaml's are not guarded by a flag: adding
     to the aliases of a canonical element a name that is there, and
     demoting a constant. The functions that can meet one answer an
     option, and [None] is the fatal error. The checks that the OCaml
     makes only under a flag ([invariant], [invariant_add_result], and
     assertions) are not made by the model's functions; the first two
     are functions of their own, [Aliases.invariant] and
     [Aliases.invariant_add_result], that answer whether the check
     passes.
   - [Not_found] is [None].
   - Where several aliases are equally early, [find_earliest_alias]
     takes in the OCaml the one that an order on names
     ([Name.compare]) puts first, and here the first of a list. The
     model has no such order; there is at most one earliest alias when
     the variables have different binding times.
   - [print] is not modelled.

   What is stated of [Aliases.add], which is given two canonical
   elements found to be equal:
   - what it answers ([add_result_statement]): the canonical element of
     the merged class and the name demoted are its two arguments, the
     later of them demoted; bottom exactly for two different constants
     ([add_bottom_statement]); neither of the model's two fatal errors
     for two different canonical elements of a consistent structure
     ([add_defined_statement]), which says nothing of the compiler's
     resolver, total here; of a canonical name and a constant, the name
     is demoted ([add_name_constant_statement]);
   - the classes afterwards ([add_canonical_element_statement],
     [add_aliased_statement]): the two classes are merged and the
     others are as they were; what was canonical is, but for the name
     demoted ([add_is_canonical_statement]); what was the same on each
     class and on the two arguments is the same on each class
     ([add_classes_uniform_statement]);
   - an environment that respects the structure and gives the two the
     same value respects the result ([add_respected_statement]), and
     conversely ([add_respected_equal_statement],
     [add_respected_before_statement]);
   - the invariant is kept, part by part: consistency always, the name
     modes recorded always, the order only if one of the two arguments
     is earlier than the other ([add_earliest_statement], and the
     question there).
   And of [Aliases.get_aliases]: a simple is in the set exactly when it
   is known to be equal to the element asked about
   ([get_aliases_statement]). The other functions on sets of aliases
   and [Aliases.add_alias_set] are stated in
   FlambdaEnvAliasesSetSpec.v.

   [aliases_well_formed] is the invariant that the statements use. It
   asks more than the compiler checks, and each of its parts says so
   where it is defined. FlambdaEnvAliasesInvariantSpec.v has it as a
   test, and says what [Aliases.invariant] and
   [Aliases.invariant_add_result], the model's copies of the checks that
   the OCaml makes under a flag, answer for a structure that has it.
   FlambdaEnvAliasesRenamingSpec.v has the renaming of the structure,
   its free variables, and that the functions commute with renaming.

   The proofs are in FlambdaEnvAliasesProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The classes                                                              *)
(* ======================================================================== *)

(* [simple_equal] tells whether two simples are the same: the test to use
   where simples are keys. *)
Definition simple_equal_decides_equality_statement : Prop :=
  decides_equality simple_equal.

(* The structure has demoted the name [n] to [canonical_element]. *)
Definition demoted_to (t : Aliases.t) (n : name) (canonical_element : simple)
    : Prop :=
  Keyed.lookup name_eqb n (Aliases.canonical_elements t)
  = Some canonical_element.

(* The structure lists the name [n] among the aliases of
   [canonical_element]. *)
Definition listed_alias (t : Aliases.t) (canonical_element : simple)
    (n : name) : Prop :=
  Aliases.Aliases_of_canonical_element.mem
    (Aliases.get_aliases_of_canonical_element t canonical_element) n
  = true.

(* A simple that the structure has not demoted: a constant, or a name
   that is alone in its class or is the canonical element of it. *)
Definition is_canonical (t : Aliases.t) (e : simple) : Prop :=
  Aliases.canonical t e = Aliases.Is_canonical.

(* The canonical element of the class of a simple. *)
Definition canonical_element_of (t : Aliases.t) (e : simple) : simple :=
  match Aliases.canonical t e with
  | Aliases.Is_canonical => e
  | Aliases.Alias_of_canonical canonical_element => canonical_element
  end.

(* Two simples are known to be equal: they are in one class. *)
Definition aliased (t : Aliases.t) (e e' : simple) : Prop :=
  canonical_element_of t e = canonical_element_of t e'.

(* The names that the structure speaks of. *)
Definition aliases_mention (t : Aliases.t) (n : name) : Prop :=
  (∃ canonical_element : simple, demoted_to t n canonical_element)
  ∨ (∃ n' : name, demoted_to t n' (simple_of_name n))
  ∨ (∃ canonical_element : simple, listed_alias t canonical_element n)
  ∨ (∃ n' : name, listed_alias t (simple_of_name n) n').

(* ======================================================================== *)
(* What the structure means                                                 *)
(* ======================================================================== *)

(* Every name that has been demoted has the value of its canonical
   element. *)
Definition canonical_elements_respected (t : Aliases.t) (vars : env)
    : Prop :=
  ∀ (n : name) (canonical_element : simple),
    demoted_to t n canonical_element ->
    value_of_name vars n = eval_simple vars canonical_element.

(* Every name listed among the aliases of a canonical element has its
   value. *)
Definition listed_aliases_respected (t : Aliases.t) (vars : env) : Prop :=
  ∀ (canonical_element : simple) (n : name),
    listed_alias t canonical_element n ->
    value_of_name vars n = eval_simple vars canonical_element.

(* An environment of the machine respects the structure. *)
Definition aliases_respected (t : Aliases.t) (vars : env) : Prop :=
  canonical_elements_respected t vars ∧ listed_aliases_respected t vars.

(* Every environment respects the structure that knows nothing. *)
Definition empty_respected_statement : Prop :=
  ∀ vars : env, aliases_respected Aliases.empty vars.

(* Simples that are known to be equal have the same value. *)
Definition respected_aliased_statement : Prop :=
  ∀ (t : Aliases.t) (vars : env) (e e' : simple),
    aliases_respected t vars ->
    aliased t e e' ->
    eval_simple vars e = eval_simple vars e'.

(* What the two queries answer has the value of what they were asked
   about. Nothing is asked of the structure, of the binding times or of
   the name modes. *)
Definition respected_canonical_statement : Prop :=
  ∀ (t : Aliases.t) (vars : env) (n : name),
    aliases_respected t vars ->
    eval_simple vars (Aliases.get_canonical_ignoring_name_mode t n)
    = value_of_name vars n.

Definition respected_canonical_at_mode_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (vars : env) (element : simple)
    (elt_name_mode min_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t) (answer : simple),
    aliases_respected t vars ->
    Aliases.get_canonical_element_exn binding_time_resolver
      binding_times_and_modes t element elt_name_mode min_name_mode
      min_binding_time
    = Some answer ->
    eval_simple vars answer = eval_simple vars element.

(* Respect depends on the environment only through the values of the
   names that the structure speaks of. So binding a variable that it does
   not speak of keeps respect. *)
Definition respected_coincide_statement : Prop :=
  ∀ (t : Aliases.t) (vars vars' : env),
    (∀ n : name,
      aliases_mention t n -> value_of_name vars n = value_of_name vars' n) ->
    aliases_respected t vars ->
    aliases_respected t vars'.

(* Alias types. A value has the alias type of a name when it is the
   value of the name (FlambdaTypeMeaningSpec.alias_satisfied_statement).
   In an environment that respects the structure, the alias type of a
   name and that of its canonical element are the same type: the same
   values have them. *)
Definition respected_alias_type_statement : Prop :=
  ∀ (t : Aliases.t) (statics : static_env) (vars : env) (v : value)
    (k : flambda_kind) (n : name),
    aliases_respected t vars ->
    (type_satisfied statics vars v (alias_type_of k (simple_of_name n))
     <-> type_satisfied statics vars v
           (alias_type_of k (Aliases.get_canonical_ignoring_name_mode t n))).

(* ======================================================================== *)
(* The order                                                                *)
(* ======================================================================== *)

(* On two names, the order on simples is the order on names. *)
Definition defined_earlier_names_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (alias than : name),
    Aliases.defined_earlier binding_time_resolver binding_times_and_modes
      (simple_of_name alias) (simple_of_name than)
    = Aliases.name_defined_earlier binding_time_resolver
        binding_times_and_modes alias than.

(* A constant is earlier than every name, and nothing is earlier than a
   constant. *)
Definition constant_earliest_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (c : const) (e : simple) (n : name),
    Aliases.defined_earlier binding_time_resolver binding_times_and_modes
      (Const c) (simple_of_name n)
    = true
    ∧ Aliases.defined_earlier binding_time_resolver binding_times_and_modes
        e (Const c)
      = false.

(* A symbol is earlier than every variable, and nothing but a constant is
   earlier than a symbol: no variable, and no other symbol. *)
Definition symbol_earlier_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (s : symbol) (x : variable) (n : name),
    Aliases.name_defined_earlier binding_time_resolver
      binding_times_and_modes (Name_symbol s) (Name_var x)
    = true
    ∧ Aliases.name_defined_earlier binding_time_resolver
        binding_times_and_modes n (Name_symbol s)
      = false.

(* The binding time that the typing environment gives a variable, which
   for a variable that it has no entry for is that of the variables of
   other compilation units; and the one that the resolver gives. *)
Definition binding_time_in {A : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (x : variable) : Binding_time.t :=
  Binding_time.With_name_mode.binding_time
    (Aliases.binding_time_and_name_mode binding_times_and_modes (Var x)).

Definition resolved_binding_time
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (x : variable) : Binding_time.t :=
  Binding_time.With_name_mode.binding_time
    (binding_time_resolver (Name_var x)).

(* Of two variables, the earlier is the one with the lesser binding time
   in the typing environment; where the two have the same, the one with
   the lesser binding time by the resolver. *)

(* CR mshinwell: two different variables with the same binding time in
   the typing environment are ordered here by what the resolver says,
   and are not ordered if it says the same of both. The compiler has no
   order there: the resolver is asked only for variables of one
   compilation unit, and for the unit being compiled it stops with a
   fatal error ([Typing_env.binding_time_resolver],
   [Compilenv.get_unit]). So the compiler relies on the variables of
   the unit being compiled having different binding times, which
   nothing there states or checks. The model asks for it: that
   different variables have different binding times is a part of the
   invariant of the typing environment, which the Spec file of the
   typing environment states, and there the model is stricter than the
   compiler (decision D32). It is not certain that the compiler's
   environments have it: a comment in [Typing_env_level.concat] says
   that a join may give several variables one binding time.
   Discrepancy T3-1. *)
Definition variable_earlier_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (x y : variable),
    Aliases.name_defined_earlier binding_time_resolver
      binding_times_and_modes (Name_var x) (Name_var y)
    = true
    <-> (binding_time_in binding_times_and_modes x
         < binding_time_in binding_times_and_modes y
         ∨ (binding_time_in binding_times_and_modes x
            = binding_time_in binding_times_and_modes y
            ∧ resolved_binding_time binding_time_resolver x
              < resolved_binding_time binding_time_resolver y)).

(* The order is strict: nothing is earlier than itself, and it is
   transitive. *)
Definition defined_earlier_strict_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (e e' e'' : simple),
    Aliases.defined_earlier binding_time_resolver binding_times_and_modes
      e e
    = false
    ∧ (Aliases.defined_earlier binding_time_resolver binding_times_and_modes
         e e'
       = true ->
       Aliases.defined_earlier binding_time_resolver binding_times_and_modes
         e' e''
       = true ->
       Aliases.defined_earlier binding_time_resolver binding_times_and_modes
         e e''
       = true).

(* ======================================================================== *)
(* The canonical element                                                    *)
(* ======================================================================== *)

(* The three maps describe the same classes. First, what a name has been
   demoted to has not been demoted itself... *)

(* The model is stricter than the compiler here (decision D32). The
   compiler's [invariant], which runs only with heavy invariant checks,
   reads the two inverse maps and never [canonical_elements]: nothing in
   the compiler checks [canonical_elements_canonical] or
   [aliases_listed], the two parts of [aliases_consistent]. [add] keeps
   them ([add_consistent_statement]). Discrepancy T3-3. *)
Definition canonical_elements_canonical (t : Aliases.t) : Prop :=
  ∀ (n : name) (canonical_element : simple),
    demoted_to t n canonical_element -> is_canonical t canonical_element.

(* ... and the names listed among the aliases of a canonical element are
   those that have been demoted to it. *)
Definition aliases_listed (t : Aliases.t) : Prop :=
  ∀ (n : name) (canonical_element : simple),
    demoted_to t n canonical_element <-> listed_alias t canonical_element n.

Definition aliases_consistent (t : Aliases.t) : Prop :=
  canonical_elements_canonical t ∧ aliases_listed t.

(* Every canonical element is earlier than the names demoted to it. The
   compiler checks this of the canonical elements that are names, in
   [invariant] and [invariant_add_result], with heavy invariant checks
   only; FlambdaEnvAliasesInvariantSpec.v says how those checks and this
   condition are related. *)
Definition canonical_elements_earliest {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) : Prop :=
  ∀ (n : name) (canonical_element : simple),
    demoted_to t n canonical_element ->
    Aliases.defined_earlier binding_time_resolver binding_times_and_modes
      canonical_element (simple_of_name n)
    = true.

(* The name mode recorded for an alias is the one the typing environment
   gives it. The model is stricter than the compiler here (decision
   D32): the compiler records the name mode when the name is demoted
   ([add_alias_between_canonical_elements]), reads it later
   ([find_earliest_candidates]), and nothing there checks that it is
   still the name mode of the name. [add] keeps the condition
   ([add_name_modes_recorded_statement]) while the typing environment
   does not change the name mode of a name that the structure speaks of
   ([name_modes_recorded_coincide_statement]). Discrepancy T3-3. *)
Definition name_modes_recorded {A : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) : Prop :=
  ∀ (canonical_element : simple) (n : name) (name_mode : Name_mode.t),
    Keyed.lookup name_eqb n
      (Aliases.get_aliases_of_canonical_element t canonical_element)
    = Some name_mode ->
    name_mode
    = Aliases.compute_name_mode_unscoped binding_times_and_modes
        (simple_of_name n).

(* The invariant of the structure, for a typing environment. *)
Definition aliases_well_formed {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) : Prop :=
  aliases_consistent t
  ∧ canonical_elements_earliest binding_time_resolver
      binding_times_and_modes t
  ∧ name_modes_recorded binding_times_and_modes t.

(* [Aliases.get_canonical_ignoring_name_mode] answers the canonical
   element of the class of the name. *)
Definition get_canonical_ignoring_name_mode_statement : Prop :=
  ∀ (t : Aliases.t) (n : name),
    Aliases.get_canonical_ignoring_name_mode t n
    = canonical_element_of t (simple_of_name n).

(* The canonical element of a class is not demoted: it is its own
   canonical element. *)
Definition canonical_element_canonical_statement : Prop :=
  ∀ (t : Aliases.t) (e : simple),
    aliases_consistent t -> is_canonical t (canonical_element_of t e).

Definition canonical_element_idempotent_statement : Prop :=
  ∀ (t : Aliases.t) (e : simple),
    aliases_consistent t ->
    canonical_element_of t (canonical_element_of t e)
    = canonical_element_of t e.

(* If it is a name, it is the name asked about or one that the structure
   speaks of. *)
Definition canonical_element_mentioned_statement : Prop :=
  ∀ (t : Aliases.t) (n n' : name),
    Aliases.get_canonical_ignoring_name_mode t n = simple_of_name n' ->
    n' = n ∨ aliases_mention t n'.

(* A name that the structure does not speak of is its own canonical
   element. *)
Definition canonical_element_unmentioned_statement : Prop :=
  ∀ (t : Aliases.t) (n : name),
    ~ aliases_mention t n ->
    Aliases.get_canonical_ignoring_name_mode t n = simple_of_name n.

(* The canonical element of a class is earlier than every other element
   of the class. *)
Definition canonical_element_earliest_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (e : simple),
    canonical_elements_earliest binding_time_resolver
      binding_times_and_modes t ->
    canonical_element_of t e <> e ->
    Aliases.defined_earlier binding_time_resolver binding_times_and_modes
      (canonical_element_of t e) e
    = true.

(* So the classes determine the canonical elements: two structures that
   have the same classes, in each of which the canonical element of a
   class is earlier than its other elements, have the same canonical
   elements. Where that holds it does not matter in which order the
   aliases were added. It holds of what [Aliases.add] builds unless two
   symbols, or two variables bound at one time, come into one class
   ([add_earliest_statement]). *)
Definition canonical_element_determined_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t t' : Aliases.t),
    aliases_consistent t ->
    aliases_consistent t' ->
    canonical_elements_earliest binding_time_resolver
      binding_times_and_modes t ->
    canonical_elements_earliest binding_time_resolver
      binding_times_and_modes t' ->
    (∀ e e' : simple, aliased t e e' <-> aliased t' e e') ->
    ∀ e : simple, canonical_element_of t e = canonical_element_of t' e.

(* It is the constant of the class, if the class has one. *)
Definition canonical_element_constant_statement : Prop :=
  ∀ (t : Aliases.t) (e : simple) (c : const),
    aliased t e (Const c) -> canonical_element_of t e = Const c.

(* If the class has a symbol, it is that symbol or a constant: not a
   variable, and no other symbol. *)
Definition canonical_element_symbol_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (e : simple) (s : symbol),
    canonical_elements_earliest binding_time_resolver
      binding_times_and_modes t ->
    aliased t e (Symbol s) ->
    canonical_element_of t e = Symbol s
    ∨ (∃ c : const, canonical_element_of t e = Const c).

(* ======================================================================== *)
(* The canonical element at a name mode                                     *)
(* ======================================================================== *)

(* A name of mode [name_mode] may be used where [min_name_mode] is asked
   for. *)
Definition name_mode_at_least (name_mode min_name_mode : Name_mode.t)
    : Prop :=
  ∃ c : Z,
    Name_mode.compare_partial_order name_mode min_name_mode = Some c
    ∧ 0 <= c.

(* The name mode that [Aliases.get_canonical_element_exn] takes the
   canonical element of [element] to have: the one its caller gives for
   [element], if that is the canonical element; otherwise the one of the
   typing environment where the variables bound at [min_binding_time] or
   later are in scope. *)
Definition canonical_name_mode {A : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (element : simple) (elt_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t) : Name_mode.t :=
  match Aliases.canonical t element with
  | Aliases.Is_canonical => elt_name_mode
  | Aliases.Alias_of_canonical canonical_element =>
    Aliases.compute_name_mode binding_times_and_modes canonical_element
      min_binding_time
  end.

(* Where the canonical element may be used, it is the answer. *)
Definition canonical_at_mode_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (element : simple)
    (elt_name_mode min_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t),
    name_mode_at_least
      (canonical_name_mode binding_times_and_modes t element elt_name_mode
         min_binding_time)
      min_name_mode ->
    Aliases.get_canonical_element_exn binding_time_resolver
      binding_times_and_modes t element elt_name_mode min_name_mode
      min_binding_time
    = Some (canonical_element_of t element).

(* Whatever the answer, it may be used: it is the element asked about,
   which its caller says may be, or its name mode in the typing
   environment, in the scope given, is at least the one asked for. *)
Definition canonical_at_mode_name_mode_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (element : simple)
    (elt_name_mode min_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t) (answer : simple),
    Aliases.get_canonical_element_exn binding_time_resolver
      binding_times_and_modes t element elt_name_mode min_name_mode
      min_binding_time
    = Some answer ->
    (answer = element ∧ name_mode_at_least elt_name_mode min_name_mode)
    ∨ name_mode_at_least
        (Aliases.compute_name_mode binding_times_and_modes answer
           min_binding_time)
        min_name_mode.

(* And it is in the class of the element asked about. *)
Definition canonical_at_mode_aliased_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (element : simple)
    (elt_name_mode min_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t) (answer : simple),
    aliases_consistent t ->
    Aliases.get_canonical_element_exn binding_time_resolver
      binding_times_and_modes t element elt_name_mode min_name_mode
      min_binding_time
    = Some answer ->
    aliased t answer element.

(* An alias of [canonical_element] that may be used at the name mode
   [order]: it was recorded with that name mode when it was demoted and,
   unless that is [In_types], it still has it in the scope given. *)
Definition alias_candidate {A : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element : simple)
    (min_binding_time : Binding_time.t) (order : Name_mode.t) (n : name)
    : Prop :=
  Keyed.lookup name_eqb n
    (Aliases.get_aliases_of_canonical_element t canonical_element)
  = Some order
  ∧ (order = Name_mode.In_types
     ∨ Aliases.compute_name_mode binding_times_and_modes (simple_of_name n)
         min_binding_time
       = order).

(* Where the canonical element may not be used, the answer is an alias
   that may: one of the least name mode that will do and has such an
   alias, and no other alias of that name mode is earlier than it. (So
   asked for [Phantom], an alias recorded as [Phantom] is preferred to
   one recorded as [Normal], even an earlier one.) *)
Definition earliest_alias_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (element : simple)
    (elt_name_mode min_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t) (answer : simple),
    ~ name_mode_at_least
        (canonical_name_mode binding_times_and_modes t element elt_name_mode
           min_binding_time)
        min_name_mode ->
    Aliases.get_canonical_element_exn binding_time_resolver
      binding_times_and_modes t element elt_name_mode min_name_mode
      min_binding_time
    = Some answer ->
    ∃ (n : name) (order : Name_mode.t),
      answer = simple_of_name n
      ∧ name_mode_at_least order min_name_mode
      ∧ alias_candidate binding_times_and_modes t
          (canonical_element_of t element) min_binding_time order n
      ∧ ∀ (n' : name) (order' : Name_mode.t),
          name_mode_at_least order' min_name_mode ->
          alias_candidate binding_times_and_modes t
            (canonical_element_of t element) min_binding_time order' n' ->
          name_mode_at_least order' order
          ∧ (order' = order ->
             Aliases.name_defined_earlier binding_time_resolver
               binding_times_and_modes n' n
             = false).

(* And there is no answer only where no alias may be used. *)
Definition no_alias_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (element : simple)
    (elt_name_mode min_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t),
    ~ name_mode_at_least
        (canonical_name_mode binding_times_and_modes t element elt_name_mode
           min_binding_time)
        min_name_mode ->
    (Aliases.get_canonical_element_exn binding_time_resolver
       binding_times_and_modes t element elt_name_mode min_name_mode
       min_binding_time
     = None
     <-> ∀ (n : name) (order : Name_mode.t),
           name_mode_at_least order min_name_mode ->
           ~ alias_candidate binding_times_and_modes t
               (canonical_element_of t element) min_binding_time order n).

(* ======================================================================== *)
(* Adding an alias                                                          *)
(* ======================================================================== *)

(* [Aliases.add] is given two canonical elements that have been found to
   be equal. Three statements say what it answers, whatever the
   structure. First, when it succeeds: the canonical element of the
   merged class and the name that was demoted are the two arguments,
   the later of the two demoted; if neither is earlier than the other,
   the first is demoted. *)
Definition add_result_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result),
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    if Aliases.defined_earlier binding_time_resolver binding_times_and_modes
         canonical_element1 canonical_element2
    then
      Aliases.canonical_element result = canonical_element1
      ∧ simple_of_name (Aliases.demoted_name result) = canonical_element2
    else
      Aliases.canonical_element result = canonical_element2
      ∧ simple_of_name (Aliases.demoted_name result) = canonical_element1.

(* Second: the answer is bottom exactly when the two are different
   constants, which are never equal. *)
Definition add_bottom_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple),
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some Or_bottom.Bottom
    <-> (∃ c1 c2 : const,
          canonical_element1 = Const c1
          ∧ canonical_element2 = Const c2
          ∧ c1 <> c2).

(* Third: [add] answers when the structure is consistent and the two are
   canonical and different, which is what [aliases.mli] asks of the
   caller: neither of the two fatal errors that the model has ([None])
   happens. This is about the model, whose resolver is total. It does
   not say that the compiler's [add] returns: for two variables of the
   unit being compiled with one binding time, the compiler stops in the
   resolver (discrepancy T3-1). *)
Definition add_defined_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple),
    aliases_consistent t ->
    is_canonical t canonical_element1 ->
    is_canonical t canonical_element2 ->
    canonical_element1 <> canonical_element2 ->
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    <> None.

(* A canonical name and a constant: [add] answers, whichever comes
   first, and it is the name that is demoted. *)
Definition add_name_constant_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (n : name) (c : const),
    aliases_consistent t ->
    is_canonical t (simple_of_name n) ->
    (∃ result : Aliases.add_result,
      Aliases.add binding_time_resolver binding_times_and_modes t
        (simple_of_name n) (Const c)
      = Some (Or_bottom.Ok result)
      ∧ Aliases.demoted_name result = n
      ∧ Aliases.canonical_element result = Const c)
    ∧ (∃ result : Aliases.add_result,
        Aliases.add binding_time_resolver binding_times_and_modes t (Const c)
          (simple_of_name n)
        = Some (Or_bottom.Ok result)
        ∧ Aliases.demoted_name result = n
        ∧ Aliases.canonical_element result = Const c).

(* The classes after [add]: the class of the name demoted is merged into
   that of the canonical element, and every other class is as it was.
   As an equation: the canonical element of a simple is the new
   canonical element if it was the one demoted, and has not changed
   otherwise. *)
Definition add_canonical_element_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result) (e : simple),
    aliases_consistent t ->
    is_canonical t canonical_element1 ->
    is_canonical t canonical_element2 ->
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    canonical_element_of (Aliases.add_result_t result) e
    = if simple_equal (canonical_element_of t e)
           (simple_of_name (Aliases.demoted_name result))
      then Aliases.canonical_element result
      else canonical_element_of t e.

(* The same as a relation: two simples are known to be equal afterwards
   exactly when they were before, or one was known to be equal to one
   argument and the other to the other. *)
Definition add_aliased_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result) (e e' : simple),
    aliases_consistent t ->
    is_canonical t canonical_element1 ->
    is_canonical t canonical_element2 ->
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    (aliased (Aliases.add_result_t result) e e'
     <-> aliased t e e'
         ∨ (aliased t e canonical_element1 ∧ aliased t e' canonical_element2)
         ∨ (aliased t e canonical_element2 ∧ aliased t e' canonical_element1)).

(* What is canonical afterwards: what was, but for the name demoted. So
   [add] of two different canonical elements leaves one canonical name
   fewer. The two have to be different: [add] of a name and itself
   answers the structure as it was, with that name as the name demoted
   and as the canonical element. *)
Definition add_is_canonical_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result) (e : simple),
    aliases_consistent t ->
    is_canonical t canonical_element1 ->
    is_canonical t canonical_element2 ->
    canonical_element1 <> canonical_element2 ->
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    (is_canonical (Aliases.add_result_t result) e
     <-> is_canonical t e ∧ e <> simple_of_name (Aliases.demoted_name result)).

(* Whatever is the same on the simples of each class, and on the two
   arguments, is the same on the simples of each class afterwards. The
   structure has no kinds and [add] compares none; with the kind that
   the typing environment gives a simple as the property, this is how
   the classes stay of one kind. *)
Definition add_classes_uniform_statement : Prop :=
  ∀ (A B : Type)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result) (property : simple -> B),
    aliases_consistent t ->
    is_canonical t canonical_element1 ->
    is_canonical t canonical_element2 ->
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    (∀ e e' : simple, aliased t e e' -> property e = property e') ->
    property canonical_element1 = property canonical_element2 ->
    ∀ e e' : simple,
      aliased (Aliases.add_result_t result) e e' -> property e = property e'.

(* [add] keeps respect: an environment that respects the structure and
   gives the two arguments the same value respects the result. Nothing
   is asked of the structure or of the arguments. *)
Definition add_respected_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result) (vars : env),
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    aliases_respected t vars ->
    eval_simple vars canonical_element1
    = eval_simple vars canonical_element2 ->
    aliases_respected (Aliases.add_result_t result) vars.

(* Conversely, an environment that respects the result gives the two
   arguments the same value, whatever the structure was. *)
Definition add_respected_equal_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result) (vars : env),
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    aliases_respected (Aliases.add_result_t result) vars ->
    eval_simple vars canonical_element1 = eval_simple vars canonical_element2.

(* And it respects the structure as it was, if that was consistent and
   the two were canonical. So the result says what the structure said
   and that the two are equal, and nothing more. *)
Definition add_respected_before_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result) (vars : env),
    aliases_consistent t ->
    is_canonical t canonical_element1 ->
    is_canonical t canonical_element2 ->
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    aliases_respected (Aliases.add_result_t result) vars ->
    aliases_respected t vars.

(* [add] keeps the three parts of the invariant, each under what it
   needs. Consistency: the two arguments are canonical. *)
Definition add_consistent_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result),
    aliases_consistent t ->
    is_canonical t canonical_element1 ->
    is_canonical t canonical_element2 ->
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    aliases_consistent (Aliases.add_result_t result).

(* The name modes recorded: nothing more. *)
Definition add_name_modes_recorded_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result),
    name_modes_recorded binding_times_and_modes t ->
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    name_modes_recorded binding_times_and_modes (Aliases.add_result_t result).

(* The order: one of the two arguments is earlier than the other. *)
Definition comparable {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (e e' : simple) : Prop :=
  Aliases.defined_earlier binding_time_resolver binding_times_and_modes e e'
  = true
  ∨ Aliases.defined_earlier binding_time_resolver binding_times_and_modes
      e' e
    = true.

(* CR mshinwell: [add] of two canonical elements neither of which is
   earlier than the other is not said here to keep
   [canonical_elements_earliest], and does not: it demotes its first
   argument. There are two such cases, and the compiler differs on
   them.
   - Two symbols. In the compiler the same [add] succeeds, and its
     result fails [invariant] and [invariant_add_result], which are run
     only with heavy invariant checks
     (FlambdaEnvAliasesInvariantSpec.v,
     [add_two_symbols_invariant_statement]). Two different symbols are
     never equal when the program runs, so it is not certain whether
     [add] is meant never to be given two symbols, or to answer bottom
     for them, or whether [invariant] is too strong. Discrepancy T3-2;
     no decision yet.
   - Two variables with the same binding times. In the compiler [add]
     does not return for two such variables of the unit being compiled:
     the order stops with a fatal error in the resolver (the question
     at [variable_earlier_statement]; discrepancy T3-1). Here the
     resolver is total and [add] answers. The invariant of the typing
     environment rules the case out. *)
Definition add_earliest_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result),
    aliases_consistent t ->
    canonical_elements_earliest binding_time_resolver
      binding_times_and_modes t ->
    is_canonical t canonical_element1 ->
    is_canonical t canonical_element2 ->
    comparable binding_time_resolver binding_times_and_modes
      canonical_element1 canonical_element2 ->
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    canonical_elements_earliest binding_time_resolver
      binding_times_and_modes (Aliases.add_result_t result).

(* Two different simples are comparable unless they are two constants,
   two symbols, or two variables with the same binding time in the
   typing environment. The compiler relies on different variables
   having different binding times there (see the question at
   [variable_earlier_statement]); that they have is a part of the
   invariant of the typing environment, where the model is stricter
   than the compiler (decision D32). *)
Definition simples_comparable_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (e e' : simple),
    e <> e' ->
    match e, e' with
    | Const _, Const _ | Symbol _, Symbol _ => True
    | Var x, Var y =>
      binding_time_in binding_times_and_modes x
      <> binding_time_in binding_times_and_modes y ->
      comparable binding_time_resolver binding_times_and_modes e e'
    | _, _ => comparable binding_time_resolver binding_times_and_modes e e'
    end.

(* The three together: [add] keeps the invariant. The structure that
   knows nothing has it. *)
Definition add_well_formed_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result),
    aliases_well_formed binding_time_resolver binding_times_and_modes t ->
    is_canonical t canonical_element1 ->
    is_canonical t canonical_element2 ->
    comparable binding_time_resolver binding_times_and_modes
      canonical_element1 canonical_element2 ->
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    aliases_well_formed binding_time_resolver binding_times_and_modes
      (Aliases.add_result_t result).

Definition empty_well_formed_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t)),
    aliases_well_formed binding_time_resolver binding_times_and_modes
      Aliases.empty.

(* The names that the structure speaks of afterwards are among those it
   spoke of and the two arguments. *)
Definition add_mentioned_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result) (n : name),
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    aliases_mention (Aliases.add_result_t result) n ->
    aliases_mention t n
    ∨ simple_of_name n = canonical_element1
    ∨ simple_of_name n = canonical_element2.

(* The invariant reads the typing environment only through the binding
   time and the name mode that it gives the names the structure speaks
   of. So it is kept by a change of the types, and by a new entry for a
   name that the structure does not speak of. *)
Definition well_formed_coincide_statement : Prop :=
  ∀ (A A' : Type)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (binding_times_and_modes'
      : Keyed.finite_map name (A' * Binding_time.With_name_mode.t))
    (t : Aliases.t),
    (∀ n : name,
      aliases_mention t n ->
      Aliases.binding_time_and_name_mode binding_times_and_modes
        (simple_of_name n)
      = Aliases.binding_time_and_name_mode binding_times_and_modes'
          (simple_of_name n)) ->
    aliases_well_formed binding_time_resolver binding_times_and_modes t ->
    aliases_well_formed binding_time_resolver binding_times_and_modes' t.

(* The same of the name modes recorded alone, for an invariant that has
   not the order: they are kept when the typing environment gives the
   same name mode to every name that the structure speaks of. *)
Definition name_modes_recorded_coincide_statement : Prop :=
  ∀ (A A' : Type)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (binding_times_and_modes'
      : Keyed.finite_map name (A' * Binding_time.With_name_mode.t))
    (t : Aliases.t),
    (∀ n : name,
      aliases_mention t n ->
      Aliases.compute_name_mode_unscoped binding_times_and_modes
        (simple_of_name n)
      = Aliases.compute_name_mode_unscoped binding_times_and_modes'
          (simple_of_name n)) ->
    name_modes_recorded binding_times_and_modes t ->
    name_modes_recorded binding_times_and_modes' t.

(* The functions take a resolver and the typing environment's map from
   names to types and binding times, and of the map they read the
   binding times and the name modes only. So two resolvers that answer
   alike, and two maps that give every name the same binding time and
   name mode, whatever they hold beside, are the same to them. (The
   renaming of a typing environment renames the types in its map: this
   is what says that the aliases do not see that.) *)
Definition binding_times_alike {A B : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (binding_times_and_modes'
      : Keyed.finite_map name (B * Binding_time.With_name_mode.t)) : Prop :=
  ∀ n : name,
    option_map snd (Keyed.lookup name_eqb n binding_times_and_modes)
    = option_map snd (Keyed.lookup name_eqb n binding_times_and_modes').

Definition defined_earlier_alike_statement : Prop :=
  ∀ (A B : Type)
    (binding_time_resolver binding_time_resolver'
      : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (binding_times_and_modes'
      : Keyed.finite_map name (B * Binding_time.With_name_mode.t))
    (alias than : simple),
    (∀ n : name, binding_time_resolver n = binding_time_resolver' n) ->
    binding_times_alike binding_times_and_modes binding_times_and_modes' ->
    Aliases.defined_earlier binding_time_resolver binding_times_and_modes
      alias than
    = Aliases.defined_earlier binding_time_resolver' binding_times_and_modes'
        alias than.

Definition add_alike_statement : Prop :=
  ∀ (A B : Type)
    (binding_time_resolver binding_time_resolver'
      : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (binding_times_and_modes'
      : Keyed.finite_map name (B * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple),
    (∀ n : name, binding_time_resolver n = binding_time_resolver' n) ->
    binding_times_alike binding_times_and_modes binding_times_and_modes' ->
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Aliases.add binding_time_resolver' binding_times_and_modes' t
        canonical_element1 canonical_element2.

Definition get_canonical_element_exn_alike_statement : Prop :=
  ∀ (A B : Type)
    (binding_time_resolver binding_time_resolver'
      : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (binding_times_and_modes'
      : Keyed.finite_map name (B * Binding_time.With_name_mode.t))
    (t : Aliases.t) (element : simple)
    (elt_name_mode min_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t),
    (∀ n : name, binding_time_resolver n = binding_time_resolver' n) ->
    binding_times_alike binding_times_and_modes binding_times_and_modes' ->
    Aliases.get_canonical_element_exn binding_time_resolver
      binding_times_and_modes t element elt_name_mode min_name_mode
      min_binding_time
    = Aliases.get_canonical_element_exn binding_time_resolver'
        binding_times_and_modes' t element elt_name_mode min_name_mode
        min_binding_time.

(* The compiler's check, and the invariant. *)
Definition invariant_alike_statement : Prop :=
  ∀ (A B : Type)
    (binding_time_resolver binding_time_resolver'
      : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (binding_times_and_modes'
      : Keyed.finite_map name (B * Binding_time.With_name_mode.t))
    (t : Aliases.t),
    (∀ n : name, binding_time_resolver n = binding_time_resolver' n) ->
    binding_times_alike binding_times_and_modes binding_times_and_modes' ->
    Aliases.invariant binding_time_resolver binding_times_and_modes t
    = Aliases.invariant binding_time_resolver' binding_times_and_modes' t.

Definition well_formed_alike_statement : Prop :=
  ∀ (A B : Type)
    (binding_time_resolver binding_time_resolver'
      : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (binding_times_and_modes'
      : Keyed.finite_map name (B * Binding_time.With_name_mode.t))
    (t : Aliases.t),
    (∀ n : name, binding_time_resolver n = binding_time_resolver' n) ->
    binding_times_alike binding_times_and_modes binding_times_and_modes' ->
    (aliases_well_formed binding_time_resolver binding_times_and_modes t
     <-> aliases_well_formed binding_time_resolver' binding_times_and_modes'
           t).

(* ======================================================================== *)
(* The aliases of an element                                                *)
(* ======================================================================== *)

(* A simple is in a set of aliases. *)
Definition alias_set_has (set : Aliases.Alias_set.t) (e : simple) : Prop :=
  match e with
  | Var x => Name_var x ∈ Aliases.Alias_set.set_names set
  | Symbol s => Name_symbol s ∈ Aliases.Alias_set.set_names set
  | Const c => Aliases.Alias_set.set_const set = Some c
  end.

(* [Aliases.get_aliases] answers the class of the element: the simples
   known to be equal to it, itself among them. *)
Definition get_aliases_statement : Prop :=
  ∀ (t : Aliases.t) (element e : simple),
    aliases_consistent t ->
    (alias_set_has (Aliases.get_aliases t element) e <-> aliased t e element).

(* Every simple of the set has the value of the element, with nothing
   asked of the structure. *)
Definition respected_aliases_statement : Prop :=
  ∀ (t : Aliases.t) (vars : env) (element e : simple),
    aliases_respected t vars ->
    alias_set_has (Aliases.get_aliases t element) e ->
    eval_simple vars e = eval_simple vars element.
