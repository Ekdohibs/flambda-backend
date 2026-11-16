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


(* What the n-way join of typing environments guarantees.

   To be reviewed: every definition and statement of this file, against
   middle_end/flambda2/types/env/join_env.ml (the header comment, which
   states what the join assumes of its inputs, and the interface
   join_env.mli), types/join_levels.ml (the entry check
   [check_join_inputs]) and the three callers in Simplify that read the
   result: simplify/join_points.ml, simplify_switch_expr.ml and
   specialization_cost.ml. The functions themselves are transcribed in
   FlambdaJoinEnv.v and FlambdaJoinTypes.v; only the first parts of
   those exist yet. This file is the contract that the transcription is
   to be proved to meet, and that the meet (FlambdaMeetSpec.v, another
   workstream) assumes of the join it is given. Until the functions are
   there, the contract is a predicate on a function
   ([sound_join_of_environments]); the statements proved now are those
   that do not need the functions.

   The compiler has two joins. The binary one (meet_and_join.ml,
   join_levels_old.ml) is the default today; the n-way one is to become
   the default and is the one specified here. The contract does not
   mention which: any join that meets it will do, and the binary one is
   kept as an oracle for the differential tests.

   What is joined. [join_inputs]: a source environment, a scope, and a
   list of use environments, each with the identifier that the caller
   has for it. In Simplify the source is the environment in which the
   handler of a continuation would be simplified if nothing were known
   of its uses, with the parameters defined; the uses are the
   environments at the [Apply_cont]s; the scope is that of the
   continuation's definition. In the meet the source is the environment
   of the meet and the uses are the environments of the branches of a
   disjunction. The use environments are numbered from 0 in the order
   of the list ([Index]). The join answers the target environment, which
   is the source with one level more, and an analysis
   ([Analysis.t], FlambdaJoinEnv.v). Inside the join, types are joined
   from lists of pairs of an index and a type that holds in the
   environment of that index ([n_way_join_type_sound]); the section
   "The join of types, inside a join of environments" is the interface
   between the two halves of the transcription.

   What is guaranteed ([join_guarantees]), for a use and a valuation
   that satisfies the use and the source strictly: some valuation
   satisfies the target strictly, gives every variable that the use or
   the source knows what it had (the same value, or none), and gives
   every
   variable that has a definition in the analysis the value, at that
   use, of the [Simple] that the analysis says, if it has one. It is
   one valuation for the whole target: every equation of the target, on
   the names of the source and on the names that the join made, holds
   under it together. A variable of the target that neither the use nor
   the source knows may be left without a value in it; nothing here
   says whether it is.

   The decisions this rests on, and what is not promised.
   - Valuations are partial and satisfaction is strict. The n-way join
     imports a variable that only some uses define, under its own name,
     with the join of its types over those uses ([find_imported_var]);
     and makes an existential variable for a vector of [Simple]s that
     covers only some uses (the contributors of one case of a row-like
     structure). At another use no value need have that type: the
     variable means nothing there. So the valuation that witnesses the
     target at a use may leave such a variable without a value, and a
     type that reaches it by an alias is then satisfied by no value. The
     definitions for this are the typing environment's
     ([typing_env_satisfied_strictly], [env_satisfies_strictly]).
   - Poison (decision D26). Typing facts are about values that are not
     poison. The contract is stated with strict satisfaction: at a use
     that holds, a name that holds poison has a type that poison has.
     Nothing is promised of a use that is satisfied only with the
     exemption of the names that hold poison ([typing_env_satisfied]):
     a join expands aliases, and through a name that is exempt that is
     wrong (the CR at [join_admits_use]). The relations [Is_int] and
     [Get_tag] are never true of a scrutinee that holds poison, and
     [Is_null] relates poison to 0 (the decision as revised and
     amended). That is for the meaning of types to say
     ([relation_holds]); until the grammar's change is under this
     file, the [relation_holds] that it is built on still holds for
     [Is_int] and [Get_tag], with any number, of a scrutinee that holds
     poison; no statement proved of this file depends on which. The
     relations that the join rebuilds on a scrutinee say again, from
     its side, a relation that the target has for another name: they
     are true where that one is. Aliases are exact: the target's are
     respected by the witness, and a variable with a definition has
     exactly the value of its definition, poison included.
   - No leastness, and no equality with the binary join. The join that
     answers the source and says nothing meets this contract
     ([join_to_source_sound_statement]). What is promised of precision
     is stated separately, for named cases: today one,
     [join_keeps_constants].
   - Physical equality (decision D30) is not in the model. In the three
     places where the compiler's result depends on it
     ([import_or_delay_n_way_join_of_concrete_type] and
     [n_way_join_env_extension] in join_env.ml,
     [n_way_join_int_indexed_product] in meet_and_n_way_join.ml) the
     model tests an equality of types, as the compiler does under
     -flambda2-join-sharing structural. The upper bound holds of either
     branch at each; what the test gives in precision (an extension, or
     a field, that is equal in every input is kept as it is) is to be
     stated separately, with the functions.
   - That the join answers is not promised: [None] is a fatal error of
     the compiler or the model's fuel run out.
   - That the target is kinded ([typing_env_kinded]) is not part of
     [join_guarantees]: it needs more of the inputs than the upper
     bound does. It is a statement of its own, [join_keeps_kinded]: if
     the source is kinded too, and the uses have the names of the
     source at the source's kinds, the target is kinded. With it the
     contract gives the premise [uses_kinded] of a later join, and the
     kinds that a meet asks, where the target or an environment made
     from it is an input. It is to be proved of the transcription, as
     the upper bound is.
   - Code age, allocation modes and symbol projections have no meaning
     here (FlambdaTypeMeaningSpec.v, FlambdaTypingEnv.v).

   What this contract does not yet cover of Simplify's joins. It is a
   foundation for runs in which one valuation satisfies a use and the
   source strictly, through environments that are kinded; three kinds
   of input that Simplify may make are outside it, and of the
   compiler's own join it is to be proved only for the runs of the
   paragraph after them.
   - A use environment in which a parameter that holds poison has a
     type that poison has not: variant unboxing makes these (decision
     D26, discrepancy T5-7). Such a use is not satisfied strictly, and
     nothing here says that the handler's environment holds after it.
     The workstream on the unboxing of continuation parameters is to
     show what holds at a join that follows an unboxed continuation.
   - A source environment that holds the types of constants lifted in
     other branches (discrepancy T5-4): if the type of such a symbol
     were false on a path that did not define it, no valuation would
     satisfy the source there. A reading of the compiler says that
     what a branch lifts is static, and so true on every path (the CR
     at [join_admits_use]). The workstream on continuations is to
     show, for the inputs it gives the join, that the valuation of a
     use satisfies the source too, once the typing environment's owner
     has said where the type of a symbol may be used.
   - A use environment in which an alias is at a kind that is not the
     kind that the environment has for its target ([uses_kinded] fails;
     discrepancy T5-9). Ill-kinded code can make such environments, and
     the compiler's join does not test for them where it makes a
     variable for [Simple]s of a kind. It is a premise here, not an
     invariant claimed of the compiler's inputs: whoever states the
     kinds of typing environments is to say where it holds.

   The compiler's join, and the names that have no value. The
   contract is strict: [join_admits_use] is of every valuation that
   satisfies the use and the source strictly, and such a valuation may
   give no value to a variable that the use knows (nothing is asked of
   it; the target of a join is satisfied, at a use, by a valuation that
   gives none to the variables that the join brought in from other
   uses). That is what the meet assumes of the join it is given. The
   compiler's join has it not: it takes a variable that an environment
   has a type for to have a value there. So what is to be proved of
   the transcription is the contract for the runs that have the values
   that the join reads ([sound_join_where_values_are_read]):
   conditions that are named and are premises there.
   - [scrutinees_have_values]: for every relation that the target has
     on a variable, the variable has a value in every run in which the
     result of the relation has one. The join puts such a relation back
     on a scrutinee whenever every environment joined has a type for it
     ([recover_inverse_relations]). Nothing in the compiler establishes
     the condition (the CR there; discrepancy T5-10).
   - [joined_variables_have_values] and
     [imported_variables_have_values]: a variable of the source that
     every use has refined, and a variable of the use that the join
     brings into the target, has a value in the run. Without them the
     join of the types of such a variable may leave out the use at
     which it has no value (the join of types passes over a type that
     it finds to have no value: one that is bottom, or a head of naked
     immediates that its relations reduce to nothing), and the target
     then asks of the variable what the other uses say, or is bottom,
     on a run that exists. The first is the typing environment's own
     reading of an equation (a statement about a name that is not
     reached through a value in hand needs the premise that the name
     has a value).
   - [blocks_have_tags]: every block that a [Simple] of known
     variables evaluates to in the run has its tag in range. This one
     is not a condition on the compiler: it is a rule of well-formed
     terms (decision D23), which the states of the run of a
     well-formed unit have (FlambdaRangeSpec.v); the way from those
     states to a valuation of a typing environment is owed (the CR
     there). The join needs it where it makes a [Get_tag] relation
     from the variable that a type of variants has for the tag.
   Another thing that the compiler's join needs is asked of the inputs
   of every join, in the contract itself ([no_bottom_types], in
   [join_inputs_well_formed]; the target has it too): no name has the
   type bottom in an environment that is not bottom. Without it the
   target of the compiler's join may be bottom though no use is, where
   every use has bottom for what a new variable stands for. The
   witness of the contract gives the
   variables that the use or the source knows exactly what they had
   ([agrees_on_known]), and so satisfies the source of the next join or
   meet: this was agreed with the owner of the meet, which has the
   same condition and the same agreement. (That the compiler's join
   has exact agreement is for the runs of the conditions above; the
   condition on bottom types does not give it by itself.)
   An instance that shows that the transcription has not the strict
   contract without the first condition is in FlambdaJoinEnvSpec.v
   ([transcribed_join_not_strict_statement]). The upper bound of the join of
   types ([n_way_join_type_sound]) has no such condition: it is of a
   value in hand.
   One more condition, [added_equations_have_values], is for the join
   that the meet calls: every variable that the join adds an equation
   on has a value at the use. With it the witness gives each such
   variable a value ([sound_join_where_equations_have_values]), which
   the meet's contract for its join asks: the meet makes environment
   extensions of the levels of environments, and an extension asks
   that each of its names has a value.

   What the inputs must satisfy ([join_inputs_well_formed],
   [uses_kinded], and the source being satisfied with each use in
   [join_admits_use]), against the three hypotheses of the header of
   join_env.ml.
   - "Any name defined in the source environment is also defined in all
     the joined environments." Not required here, and not true of what
     Simplify passes: [check_join_inputs] says that the source may have
     variables in types, made when lifted constants were added to it,
     that no use has; and the second hypothesis itself says that a
     symbol lifted in one branch is not defined in the others. Nothing
     checks it. What the contract needs in its place: that the
     valuation of a use satisfies the source too (a premise of
     [join_admits_use], for each valuation: that a use entails the
     source is false even of the environments that the meet makes);
     and, for the caller to conclude of the machine,
     [terms_in_scope_at_uses] (a variable that may occur in terms in
     the source may occur in terms at each use).
   - "Any symbol defined in one of the joined environments is also
     defined in the source environment." Required
     ([symbols_of_uses_in_source]), for the target to be well formed
     only. [check_join_inputs] checks it of the set of symbols that
     Simplify passes beside the environments, not of the environments;
     the meet's calls are not checked.
   - "The relative order of variables defined in the source environment
     is preserved across all the joined environments."
     [binding_times_coherent]. Nothing checks it. It is not required
     for the guarantee: the compiler uses it to find the only source
     variable that can stand for a vector of [Simple]s, and then checks
     that it does ([equal_in_all_joined_envs]). It would be a premise
     of a statement that aliases between source variables are kept,
     which is not promised yet.
   Each difference is recorded as a discrepancy (T5-1 to T5-3).

   The failure that was found, and repaired. The n-way join of two
   closure types was not an upper bound where one says nothing of the
   code of a function and the other names it:
   [n_way_join_closures_entry] kept a function type that only some of
   the entries joined have ([n_way_join_function_types_before_fix] in
   FlambdaJoinTypes.v is that part of it as it was).
   [n_way_join_function_types_before_fix_unsound_statement] is
   the instance: no transcription of that function could meet
   [n_way_join_type_sound], and the contract was not weakened for it.
   The compiler has since been changed (on the integration branch) to
   keep a function type only if every entry has one;
   [n_way_join_function_types] is that, and with it a closures entry
   that holds with the function types of one of the entries joined
   holds with those joined ([n_way_join_function_types_sound_statement]).
   Discrepancies T1-4 and R3-7; a program that the compiler miscompiled
   because of it has been shown. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvKindedSpec FlambdaJoinEnv FlambdaJoinTypes.
Import ListNotations.

(* ======================================================================== *)
(* What is joined                                                           *)
(* ======================================================================== *)

(* An environment knows a variable: the variable is bound there,
   whatever its name mode and whether or not it is still in scope. This
   is the compiler's [Typing_env.mem] with no minimal name mode. *)
Definition knows (tenv : Typing_env.t) (x : variable) : Prop :=
  Typing_env.mem None tenv (Name_var x) = true.

(* The inputs of a join: the source environment, the scope after which
   the use environments are cut, and the use environments, each with
   the caller's identifier for it. *)
Record join_inputs (A : Type) : Type := {
  source : Typing_env.t;
  cut_after : Scope.t;
  uses : list (A * Typing_env.t)
}.

Arguments source {A} _.
Arguments cut_after {A} _.
Arguments uses {A} _.

(* The use environment of an index. *)
Definition use_at {A : Type} (inputs : join_inputs A) (index : Index.t)
    (use : Typing_env.t) : Prop :=
  ∃ id : A, nth_error (uses inputs) index = Some (id, use).

(* A join of environments, as a function: the variables that the names
   it makes must avoid, and the inputs. It answers the target
   environment and the analysis; [None] is a fatal error of the
   compiler, or fuel run out. *)
Definition join_of_environments (A : Type) : Type :=
  list variable -> join_inputs A -> option (Typing_env.t * Analysis.t A).

(* ======================================================================== *)
(* One valuation for the target at a use                                    *)
(* ======================================================================== *)

(* The second valuation gives every variable that the environment
   knows what the first gives it: the same value, or none in both. So
   the second satisfies every environment that the first satisfies and
   whose variables this one knows; that is what lets a caller go from
   one join, or meet, to the next with a valuation that satisfies the
   source as well as the use. (The compiler's join can force a value on
   a name that has none at a use, where the join of types leaves that
   use out. So it has this only on the runs of
   [sound_join_where_values_are_read]; [no_bottom_types] alone does not
   give it.) *)
Definition agrees_on_known (tenv : Typing_env.t) (valuation valuation' : env)
    : Prop :=
  ∀ x : variable, knows tenv x -> lookup x valuation' = lookup x valuation.

(* The second valuation gives every variable that has a definition at
   the use of index [index] the value that that definition has in the
   first, a valuation of that use, if it has one there. *)
Definition definitions_hold
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (valuation valuation' : env) : Prop :=
  ∀ (x : variable) (simples : Simples_in_joined_envs.t) (kind : flambda_kind)
    (simple : simple) (v : value),
    lookup x definitions = Some (simples, kind) ->
    Simples_in_joined_envs.find index simples = Some simple ->
    eval_simple valuation simple = Some v ->
    lookup x valuation' = Some v.

(* The target admits a use, of index [index]: every valuation that
   satisfies the use and the source extends to one that satisfies the
   target, under which the definitions of the two maps of the analysis
   hold. *)
(* CR mshinwell: the use must be satisfied strictly: a name that holds
   poison there must have a type that poison has. The use environments
   that variant unboxing makes are satisfied only with the exemption of
   the names that hold poison (decision D26), and of those nothing is
   promised here. The statement with the exemption on both sides is
   false: if [x : block of one field (= y)] and [y : {7}] at a use where
   [y] holds poison, and [x : block of one field {5}] at another, the
   join answers [x : block of one field {5, 7}], which the value of [x]
   at the first use has not. The compiler does the same, relying on a
   parameter that holds poison never being read; nothing checks that.
   So this contract does not cover the joins that follow an unboxed
   continuation: the workstream on the unboxing of continuation
   parameters owes the statement of what holds there (for instance
   that the environment tells which names may hold poison, and that the
   join leaves their types alone). Decision D26; discrepancy T5-7. *)
(* CR mshinwell: the valuation must satisfy the source too. That is a
   premise for each valuation, and about what the environments mean;
   the header of join_env.ml has one about the names they define, which
   is false of Simplify's inputs. It cannot be the premise that a use
   entails the source: an environment made from another by adding an
   equation does not entail it (the meet may forget what tied two
   names), though the valuation at hand satisfies both. Whether it
   holds of Simplify's joins is to be shown: the source has the types
   of the constants lifted while the body of the continuation was
   simplified, and if such a constant had a field computed in one
   branch, its type would not be true on a path through another. A
   reading of the compiler says that this does not happen: a constant
   with a field computed at run time is bound only at the top level of
   the unit (simplify/simplify_let_expr.ml, the test of
   [at_unit_toplevel], which a handler keeps only if its body can
   leave only through it: simplify/simplify_let_cont_expr.ml), and
   what is lifted in a branch is static, so its type is true on every
   path; small programs compiled with -O3 agree. That is a reading,
   not a statement of the model. Two owe it: the owner of the typing
   environment, to say where the type of a symbol may be used; and
   the workstream on continuations, to show it of the inputs it gives
   the join. Discrepancy T5-4. *)
Definition join_admits_use (source use target : Typing_env.t)
    (definitions definitions_at_normal_mode
       : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) : Prop :=
  ∀ (statics : static_env) (valuation : env),
    typing_env_satisfied_strictly statics valuation use ->
    typing_env_satisfied_strictly statics valuation source ->
    ∃ valuation' : env,
      typing_env_satisfied_strictly statics valuation' target
      ∧ agrees_on_known use valuation valuation'
      ∧ agrees_on_known source valuation valuation'
      ∧ definitions_hold definitions index valuation valuation'
      ∧ definitions_hold definitions_at_normal_mode index valuation
          valuation'.

(* The same of the valuations that have a property. *)
Definition join_admits_use_on (run : static_env -> env -> Prop)
    (source use target : Typing_env.t)
    (definitions definitions_at_normal_mode
       : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) : Prop :=
  ∀ (statics : static_env) (valuation : env),
    typing_env_satisfied_strictly statics valuation use ->
    typing_env_satisfied_strictly statics valuation source ->
    run statics valuation ->
    ∃ valuation' : env,
      typing_env_satisfied_strictly statics valuation' target
      ∧ agrees_on_known use valuation valuation'
      ∧ agrees_on_known source valuation valuation'
      ∧ definitions_hold definitions index valuation valuation'
      ∧ definitions_hold definitions_at_normal_mode index valuation
          valuation'.

(* The machine satisfies a use and the source together: it binds the
   variables of terms of the use, and one valuation satisfies both
   strictly and gives those variables the values that the machine has
   for them. *)
Definition use_and_source_satisfied (source use : Typing_env.t)
    (statics : static_env) (vars : env) : Prop :=
  bound_in_terms use vars
  ∧ ∃ valuation : env,
      typing_env_satisfied_strictly statics valuation use
      ∧ typing_env_satisfied_strictly statics valuation source
      ∧ agrees_in_terms use valuation vars.

(* ======================================================================== *)
(* What the inputs must satisfy                                             *)
(* ======================================================================== *)

(* One variable is bound strictly before another in an environment. *)
Definition bound_strictly_earlier (tenv : Typing_env.t) (x y : variable)
    : Prop :=
  ∃ time_x time_y : Binding_time.With_name_mode.t,
    Typing_env.binding_time_and_mode tenv (Name_var x) = Some time_x
    ∧ Typing_env.binding_time_and_mode tenv (Name_var y) = Some time_y
    ∧ Binding_time.strictly_earlier
        (Binding_time.With_name_mode.binding_time time_x)
        (Binding_time.With_name_mode.binding_time time_y)
      = true.

(* A variable that may occur in terms in the source may occur in terms
   at each use. The join does not need it: it is what the caller needs
   to conclude of the machine ([join_guarantees_in_machine_statement]),
   and it is what is left, for the variables of terms, of the first
   hypothesis of the header of join_env.ml. *)
Definition terms_in_scope_at_uses {A : Type} (inputs : join_inputs A)
    : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t) (x : variable),
    use_at inputs index use ->
    in_scope_in_terms (source inputs) x ->
    in_scope_in_terms use x.

(* The third hypothesis of the header of join_env.ml: the variables of
   the source are bound in the same order at every use that knows them.
   It is not part of [join_inputs_well_formed]. *)
Definition binding_times_coherent {A : Type} (inputs : join_inputs A)
    : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t) (x y : variable),
    use_at inputs index use ->
    knows use x ->
    knows use y ->
    bound_strictly_earlier (source inputs) x y ->
    bound_strictly_earlier use x y.

Section Contract.

(* The invariant of a typing environment (binding times that differ,
   names that are bound where they are mentioned, levels that agree
   with the types), which the typing environment's file defines. Kinds
   are not part of it: [uses_kinded] is a premise of its own. *)
Context (typing_env_well_formed : Typing_env.t -> Prop).

(* Every environment given is well formed. *)
Definition environments_well_formed {A : Type} (inputs : join_inputs A)
    : Prop :=
  typing_env_well_formed (source inputs)
  ∧ ∀ (index : Index.t) (use : Typing_env.t),
      use_at inputs index use -> typing_env_well_formed use.

(* Every use environment that is not bottom can be cut after the
   scope. *)
Definition cuts_defined {A : Type} (inputs : join_inputs A) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t),
    use_at inputs index use ->
    Typing_env.is_bottom use = false ->
    ∃ level : Typing_env_level.t,
      Typing_env.cut use (cut_after inputs) = Some level.

(* A symbol that a use defines is defined in the source. *)
Definition symbols_of_uses_in_source {A : Type} (inputs : join_inputs A)
    : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t) (s : symbol),
    use_at inputs index use ->
    Typing_env.mem None use (Name_symbol s) = true ->
    Typing_env.mem None (source inputs) (Name_symbol s) = true.

(* The variables to avoid include every variable that the source or a
   use knows. *)
Definition avoids_known {A : Type} (avoid : list variable)
    (inputs : join_inputs A) : Prop :=
  (∀ x : variable, knows (source inputs) x -> among x avoid = true)
  ∧ ∀ (index : Index.t) (use : Typing_env.t) (x : variable),
      use_at inputs index use -> knows use x -> among x avoid = true.

(* No name has the type bottom, unless the environment is bottom. *)
(* CR mshinwell: the model is stricter than the compiler here. A name
   with the type bottom has no value; the join of types leaves a bottom
   type out, and the target could then be bottom though a use is not
   (where every use has bottom for what a new variable stands for).
   This is not what keeps the join from forcing a value on a name that
   has none: a type may have no value without being bottom, and the
   conditions on the runs are for that. The meet
   makes the whole environment bottom when a type becomes bottom, and
   the n-way join adds its equations with the meet's function, so the
   environments of that path are expected to have this; the binary join
   makes environments that have it not, on purpose
   ([Meet_env.add_env_extension_maybe_bottom]). It is not known of
   every environment that Simplify builds. Discrepancy T5-10. *)
Definition no_bottom_types (tenv : Typing_env.t) : Prop :=
  Typing_env.is_bottom tenv = false ->
  ∀ (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    is_obviously_bottom ty = false.

Definition environments_without_bottom_types {A : Type}
    (inputs : join_inputs A) : Prop :=
  no_bottom_types (source inputs)
  ∧ ∀ (index : Index.t) (use : Typing_env.t),
      use_at inputs index use -> no_bottom_types use.

Definition join_inputs_well_formed {A : Type} (avoid : list variable)
    (inputs : join_inputs A) : Prop :=
  environments_well_formed inputs
  ∧ environments_without_bottom_types inputs
  ∧ cuts_defined inputs
  ∧ symbols_of_uses_in_source inputs
  ∧ avoids_known avoid inputs.

(* ======================================================================== *)
(* What the join guarantees                                                 *)
(* ======================================================================== *)

(* The target is the source with more in its last level: every name
   that had a type has one, with the binding time and the name mode it
   had; the same symbols are defined; the earlier levels, the scope and
   the first binding time in scope are those of the source. *)
Definition join_keeps_source (source target : Typing_env.t) : Prop :=
  (∀ (n : name) (ty : flambda_type)
     (binding_time_and_mode : Binding_time.With_name_mode.t),
     Keyed.lookup name_eqb n (Typing_env.names_to_types source)
     = Some (ty, binding_time_and_mode) ->
     ∃ ty' : flambda_type,
       Keyed.lookup name_eqb n (Typing_env.names_to_types target)
       = Some (ty', binding_time_and_mode))
  ∧ (∀ s : symbol,
       Typing_env.mem None target (Name_symbol s)
       = Typing_env.mem None source (Name_symbol s))
  ∧ Typing_env.prev_levels target = Typing_env.prev_levels source
  ∧ Typing_env.current_scope target = Typing_env.current_scope source
  ∧ Typing_env.min_binding_time target = Typing_env.min_binding_time source.

(* A variable that the target knows and the source does not may not
   occur in terms, and either some use that is not bottom knows it (it
   was imported under its own name) or it is none of the variables to
   avoid (it was made by the join). This is said of membership only:
   which variable the join makes is not specified. *)
Definition join_introduces {A : Type} (avoid : list variable)
    (inputs : join_inputs A) (target : Typing_env.t) : Prop :=
  ∀ x : variable,
    knows target x ->
    ~ knows (source inputs) x ->
    ~ in_scope_in_terms target x
    ∧ ((∃ (index : Index.t) (use : Typing_env.t),
          use_at inputs index use
          ∧ Typing_env.is_bottom use = false
          ∧ knows use x)
       ∨ among x avoid = false).

(* Kinds. That a [Simple] is of a kind in an environment
   ([simple_of_kind]), that a type is kinded there ([type_kinded]: each
   of its aliases is at the kind that the environment has for its
   target, and its relation variables are naked immediates there) and
   that an environment is kinded ([typing_env_kinded]: every type that
   it has for a name is kinded, two [Simple]s that it knows to be
   equal are of one kind, and a symbol is of the kind of values) are
   FlambdaTypingEnvKindedSpec.v. The aliases of a type, each with the
   kind at which it stands ([aliases_with_kinds]), are
   FlambdaTypeGrammar.v. *)

(* Every use environment is kinded. *)
(* CR mshinwell: this is a premise of the contract; it is not claimed
   that the compiler's environments have it. A type does not say at
   which kind the target of an alias is bound, the well-formedness of
   types leaves it out on purpose, and ill-kinded code (which Simplify
   must accept: decisions D10 and D12) may give a name an alias at
   another kind. The compiler's join makes a variable of the kind of
   the types joined for the [Simple]s that they are aliases to, and
   does not look at their kinds. It stops later with a fatal error if
   they differ, when it looks the [Simple]s up at that kind
   ([find_simples_in_joined_envs], through [Typing_env.find]) or meets
   the type joined with that of the variable; unless the depth limit of
   the join is reached first, and then it answers an environment in
   which the variable is of one kind and stands, at a use, for a
   [Simple] of another, and an analysis that says so. The model's
   function answers what the compiler answers. Whether an environment
   that is not kinded can reach a join is not known. Discrepancy
   T5-9. *)
Definition uses_kinded {A : Type} (inputs : join_inputs A) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t),
    use_at inputs index use -> typing_env_kinded use.

(* What the analysis speaks of. Its identifiers are those of uses that
   are not bottom, by index. A definition is of a variable that the
   target has, with the kind given, at a use with an identifier, and is
   a [Simple] of that kind in that use; a definition at the normal name
   mode is moreover a [Simple] that may occur in terms at that use.
   (Simplify makes a parameter of the kind given and passes the
   [Simple] for it.) *)
Definition analysis_describes_uses {A : Type} (inputs : join_inputs A)
    (target : Typing_env.t) (analysis : Analysis.t A) : Prop :=
  (∀ (index : Index.t) (id : A),
     Keyed.lookup Index.equal index (Analysis.external_ids analysis) = Some id
     -> ∃ use : Typing_env.t,
          nth_error (uses inputs) index = Some (id, use)
          ∧ Typing_env.is_bottom use = false)
  ∧ (∀ (x : variable) (simples : Simples_in_joined_envs.t)
       (kind : flambda_kind) (index : Index.t) (simple : simple),
       lookup x (Analysis.definitions_in_joined_envs analysis)
       = Some (simples, kind) ->
       Simples_in_joined_envs.find index simples = Some simple ->
       simple_of_kind target (Var x) kind
       ∧ ∃ (id : A) (use : Typing_env.t),
           nth_error (uses inputs) index = Some (id, use)
           ∧ Keyed.lookup Index.equal index (Analysis.external_ids analysis)
             = Some id
           ∧ simple_of_kind use simple kind)
  ∧ (∀ (x : variable) (simples : Simples_in_joined_envs.t)
       (kind : flambda_kind) (index : Index.t) (simple : simple),
       lookup x (Analysis.canonical_definitions_at_normal_mode analysis)
       = Some (simples, kind) ->
       Simples_in_joined_envs.find index simples = Some simple ->
       simple_of_kind target (Var x) kind
       ∧ ∃ (id : A) (use : Typing_env.t),
           nth_error (uses inputs) index = Some (id, use)
           ∧ Keyed.lookup Index.equal index (Analysis.external_ids analysis)
             = Some id
           ∧ simple_of_kind use simple kind
           ∧ Typing_env.mem_simple (Some Name_mode.normal) use simple = true).

(* What a join guarantees of its result. *)
Definition join_guarantees {A : Type} (avoid : list variable)
    (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) : Prop :=
  typing_env_well_formed target
  ∧ no_bottom_types target
  ∧ join_keeps_source (source inputs) target
  ∧ join_introduces avoid inputs target
  ∧ analysis_describes_uses inputs target analysis
  ∧ (∀ (index : Index.t) (use : Typing_env.t),
       use_at inputs index use ->
       join_admits_use (source inputs) use target
         (Analysis.definitions_in_joined_envs analysis)
         (Analysis.canonical_definitions_at_normal_mode analysis) index).

(* The contract of a join of environments. *)
Definition sound_join_of_environments {A : Type}
    (join : join_of_environments A) : Prop :=
  ∀ (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A),
    join_inputs_well_formed avoid inputs ->
    uses_kinded inputs ->
    join avoid inputs = Some (target, analysis) ->
    join_guarantees avoid inputs target analysis.

(* What is to be proved of the compiler's join: the contract for the
   runs that have the values that the join reads. *)

(* A variable that the source, or the use of an index, knows. *)
Definition known_at_use {A : Type} (inputs : join_inputs A)
    (index : Index.t) (x : variable) : Prop :=
  knows (source inputs) x
  ∨ ∃ use : Typing_env.t, use_at inputs index use ∧ knows use x.

(* A variable has a value at a use: it is known there and the
   valuation of the use gives it one, or it has a definition there
   that has one. (The second is for the variables that the join makes,
   which the use does not know.) A value that the valuation gives to a
   variable that is not known there does not count: a valuation that
   satisfies the use and the source may give anything to such a
   variable, and what it gives says nothing of the run. Without this,
   a run in which a variable has no value would pass the conditions
   below by a valuation that gives it one all the same, which need not
   be a value that the target can keep. *)
Definition has_value_at_use (known : variable -> Prop)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (valuation : env) (x : variable) : Prop :=
  (known x ∧ ∃ v : value, lookup x valuation = Some v)
  ∨ ∃ (simples : Simples_in_joined_envs.t) (kind : flambda_kind)
      (simple : simple) (v : value),
      lookup x definitions = Some (simples, kind)
      ∧ Simples_in_joined_envs.find index simples = Some simple
      ∧ eval_simple valuation simple = Some v.

(* For every relation that the target has on a variable, at the head
   of the type of the variable that holds its result: in a run in which
   the result has a value, the scrutinee has one. *)
(* CR mshinwell: nothing in the compiler establishes this. The join
   gives a variable [r] the type "the result of [Is_int] (or [Get_tag],
   [Is_null]) on [x]" when the type that it has joined for [x] names
   [r] for that result and every environment joined has a type for
   what [x] stands for ([recover_inverse_relations] in join_env.ml,
   whose comment gives that test as what makes the relation valid).
   That type is true of a value of [r] only if [x] has a value. An
   environment may have a type for a variable that has no value in the
   run: an earlier n-way join brings a variable in from some of its
   uses, with its type, and on a run through another use it stands for
   nothing (discrepancy T5-5). If [r] has a value on such a run (at
   that use it is a constant, or a variable older than [x]), the
   target is not satisfied; and a meet that later learns the value of
   [r] reduces the type of [x] from it, and may answer bottom on a run
   that exists. Simplify can make such inputs: a program has been
   compiled wrongly because of it, with the n-way join (the arm of a
   switch on [r] that the run takes is deleted). Discrepancy T5-10. *)
Definition scrutinees_have_values (known : variable -> Prop)
    (target : Typing_env.t)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (valuation : env) : Prop :=
  ∀ (r x : variable) (relation : Relation.t)
    (head : head_of_kind_naked_immediate)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb (Name_var r) (Typing_env.names_to_types target)
    = Some (Type_naked_immediate (Ok (No_alias head)), binding_time_and_mode)
    ->
    (relation, Name_var x) ∈ inverse_relations head ->
    has_value_at_use known definitions index valuation r ->
    has_value_at_use known definitions index valuation x.

(* A variable of the source that has an equation in the level cut from
   every use that is not bottom has a value. (The join gives such a
   variable the join of those equations, by a meet with the type that
   the source has for it. Of a variable with no value at a use, whose
   type there has no value, that join may leave the use out, and the
   target then asks of the variable what the other uses say; or the
   meet may be bottom.) Whoever adds an equation on a name is to add it
   where the name has a value: that is the typing environment's reading
   of an equation, and what a caller shows this from. *)
(* CR mshinwell: what establishes this in the compiler is a discipline
   and not a check. Simplify adds an equation on a variable that a term
   has just bound or that it reaches through a value in hand, and the
   meet likewise; a join adds one on a variable of its source only if
   every use has one, and then the variable has a value by this same
   condition at that join. So it is expected of every join that
   Simplify and the meet perform, and the meet can keep it as it goes;
   nothing states it. Discrepancy T5-10. *)
Definition joined_variables_have_values {A : Type} (inputs : join_inputs A)
    (valuation : env) : Prop :=
  ∀ x : variable,
    knows (source inputs) x ->
    (∀ (index : Index.t) (use : Typing_env.t),
       use_at inputs index use ->
       Typing_env.is_bottom use = false ->
       ∃ (level : Typing_env_level.t) (ty : flambda_type),
         Typing_env.cut use (cut_after inputs) = Some level
         ∧ Keyed.lookup name_eqb (Name_var x)
             (Typing_env_level.equations level)
           = Some ty) ->
    ∃ v : value, lookup x valuation = Some v.

(* A variable that the use knows, that the source does not know, and
   that the target knows, has a value: the join brought it in, and
   gave it the join of the types that the uses that know it have for
   it. (A type that has no value is left out of that join by the join
   of types, where it sees that; and a type that has one value makes
   the variable an alias of a constant. So of a variable with no value
   at the use the target may ask a value.) *)
(* CR mshinwell: nothing in the compiler establishes this. A variable
   that a use knows and the source does not is one that the branch
   bound, which has a value, or one that an earlier join in the branch
   made or brought in, which has none on the runs that came through
   the uses that did not have it (discrepancy T5-5). A program is
   known that the compiler compiles wrongly through such a variable (a
   later join reads its type at a use where it has no value, and the
   program breaks [scrutinees_have_values]); none is known that breaks
   this condition by itself, which needs a variable that two uses know
   and the source does not. Discrepancy T5-10. *)
Definition imported_variables_have_values {A : Type}
    (inputs : join_inputs A) (target : Typing_env.t) (index : Index.t)
    (valuation : env) : Prop :=
  ∀ (use : Typing_env.t) (x : variable),
    use_at inputs index use ->
    knows use x ->
    ~ knows (source inputs) x ->
    knows target x ->
    ∃ v : value, lookup x valuation = Some v.

(* Every block that a [Simple] of known variables evaluates to in the
   run has its tag in range. As for [has_value_at_use], what the
   valuation gives to a variable that is not known says nothing of the
   run, and nothing is asked of it. *)
(* CR mshinwell: this is a rule of well-formed terms and not something
   that the compiler is to establish, or tests at a join: the machine
   makes a block of any tag, and the terms of the compiler have tags in
   range (decision D23). The machine's side of it is stated: every
   state of the run of a well-formed unit is in range
   ([well_formed_in_range_statement], FlambdaRangeSpec.v), and in such
   a state a block that a variable holds has its tag in range
   ([blocks_in_range_statement]), as has every static block
   ([statics_in_range]). What is owed is the way from there to this
   predicate, which is about a valuation of a typing environment and
   not about the variables of the machine. A valuation that represents
   the run gives the variables that may occur in terms what the
   machine has; the others, which exist only in types, have values
   that earlier joins and meets chose, and those have to be shown to
   be values of the run. And the predicate asks a tag in range of the
   static block of every symbol, mentioned by the term or not, since a
   symbol has no free variable: that is asked of the whole static
   environment. So that a term is well formed does not by itself give
   this predicate of a valuation. The join needs it
   where it makes a [Get_tag] relation from the variable that a type of
   variants has for the tag ([recover_inverse_relations]): the type
   says of a block of any tag that the variable holds its tag, and the
   relation relates a block to its tag only if the tag is in range
   (FlambdaTypeMeaningSpec.v, [relation_holds]). Without the condition
   the statement is false of the transcription, whatever the compiler
   does: a use where [x] is a block of tag 300, of a type of variants
   with no immediates whose variable [g] for the tag holds 300, is
   satisfied, and the target says of [g] that it is the [Get_tag] of
   [x], which no number is. The meet uses the same rule where a
   relation is made from a primitive. The other way to have it is for
   a type of variants to ask of its blocks a tag in range; that is for
   the owner of the meaning of types. *)
Definition blocks_have_tags (known : variable -> Prop) (statics : static_env)
    (valuation : env) : Prop :=
  ∀ (simple : simple) (v : value) (tag : Z) (fields : list value),
    (∀ x : variable, x ∈ free_variables_of_simple simple -> known x) ->
    eval_simple valuation simple = Some v ->
    block_of statics v = Some (tag, fields) ->
    is_tag tag.

(* The run has the values that the join reads, and its blocks have
   tags in range. *)
Definition values_are_read {A : Type} (inputs : join_inputs A)
    (target : Typing_env.t)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (statics : static_env) (valuation : env) : Prop :=
  scrutinees_have_values (known_at_use inputs index) target definitions
    index valuation
  ∧ joined_variables_have_values inputs valuation
  ∧ imported_variables_have_values inputs target index valuation
  ∧ blocks_have_tags (known_at_use inputs index) statics valuation.

(* What the compiler's join guarantees of its result: [join_guarantees]
   with the upper bound for those runs only. *)
Definition join_guarantees_where_values_are_read {A : Type}
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) : Prop :=
  typing_env_well_formed target
  ∧ no_bottom_types target
  ∧ join_keeps_source (source inputs) target
  ∧ join_introduces avoid inputs target
  ∧ analysis_describes_uses inputs target analysis
  ∧ (∀ (index : Index.t) (use : Typing_env.t),
       use_at inputs index use ->
       join_admits_use_on
         (values_are_read inputs target
            (Analysis.definitions_in_joined_envs analysis) index)
         (source inputs) use target
         (Analysis.definitions_in_joined_envs analysis)
         (Analysis.canonical_definitions_at_normal_mode analysis) index).

(* The contract that the transcription of the compiler's join is to be
   proved to meet. A join that meets [sound_join_of_environments] meets
   it. *)
Definition sound_join_where_values_are_read {A : Type}
    (join : join_of_environments A) : Prop :=
  ∀ (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A),
    join_inputs_well_formed avoid inputs ->
    uses_kinded inputs ->
    join avoid inputs = Some (target, analysis) ->
    join_guarantees_where_values_are_read avoid inputs target analysis.

(* One more condition on runs, and a stronger contract with it: for
   the join that the meet calls. *)

(* The equations that the target has in its last level and the source
   has not there: those that the join added. *)
Definition added_equation (source target : Typing_env.t) (x : variable)
    (ty : flambda_type) : Prop :=
  (Name_var x, ty)
  ∈ Typing_env_level.equations
      (Typing_env.One_level.level (Typing_env.current_level target))
  ∧ ~ (Name_var x, ty)
      ∈ Typing_env_level.equations
          (Typing_env.One_level.level (Typing_env.current_level source)).

(* Every variable that the join added an equation on has a value at
   the use. *)
(* CR mshinwell: nothing in the compiler establishes this. The join
   adds an equation on a variable that it makes for a vector of
   [Simple]s, which has a value at a use only if the vector has a
   [Simple] there that has one; and on a variable that it brings in
   from the uses that know it, which has no value on a run through a
   use that does not (discrepancy T5-5). The environment answered is
   satisfied on such a run all the same: a name with no value is asked
   nothing ([sound_join_where_values_are_read] does not need this
   condition, unless the meet that adds the equations does). But the
   meet cuts a level of an environment into an environment extension
   and adds it to another, and an extension asks that each of its names
   has a value of the type of its equation
   (FlambdaTypeMeaningSpec.v). So the meet's contract for the join
   that it is given asks that the join adds no equation on a name with
   no value, and the compiler's join has that only on the runs of this
   condition. Discrepancies T5-5 and T5-10. *)
Definition added_equations_have_values {A : Type} (inputs : join_inputs A)
    (target : Typing_env.t)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (valuation : env) : Prop :=
  ∀ (x : variable) (ty : flambda_type),
    added_equation (source inputs) target x ty ->
    has_value_at_use (known_at_use inputs index) definitions index valuation
      x.

(* The target admits a use, for the valuations that have a property,
   and the witness gives a value to every variable that the join added
   an equation on. *)
Definition join_admits_use_with_values (run : static_env -> env -> Prop)
    (source use target : Typing_env.t)
    (definitions definitions_at_normal_mode
       : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) : Prop :=
  ∀ (statics : static_env) (valuation : env),
    typing_env_satisfied_strictly statics valuation use ->
    typing_env_satisfied_strictly statics valuation source ->
    run statics valuation ->
    ∃ valuation' : env,
      typing_env_satisfied_strictly statics valuation' target
      ∧ agrees_on_known use valuation valuation'
      ∧ agrees_on_known source valuation valuation'
      ∧ definitions_hold definitions index valuation valuation'
      ∧ definitions_hold definitions_at_normal_mode index valuation
          valuation'
      ∧ ∀ (x : variable) (ty : flambda_type),
          added_equation source target x ty ->
          ∃ v : value, lookup x valuation' = Some v.

(* The stronger contract, for the runs that have that condition too: what
   the meet asks of the join that it calls, for the compiler's join. A
   join that meets it meets [sound_join_where_values_are_read] on those
   runs. *)
Definition sound_join_where_equations_have_values {A : Type}
    (join : join_of_environments A) : Prop :=
  ∀ (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A),
    join_inputs_well_formed avoid inputs ->
    uses_kinded inputs ->
    join avoid inputs = Some (target, analysis) ->
    typing_env_well_formed target
    ∧ no_bottom_types target
    ∧ join_keeps_source (source inputs) target
    ∧ join_introduces avoid inputs target
    ∧ analysis_describes_uses inputs target analysis
    ∧ ∀ (index : Index.t) (use : Typing_env.t),
        use_at inputs index use ->
        join_admits_use_with_values
          (fun statics valuation =>
             values_are_read inputs target
               (Analysis.definitions_in_joined_envs analysis) index statics
               valuation
             ∧ added_equations_have_values inputs target
                 (Analysis.definitions_in_joined_envs analysis) index
                 valuation)
          (source inputs) use target
          (Analysis.definitions_in_joined_envs analysis)
          (Analysis.canonical_definitions_at_normal_mode analysis) index.

(* Precision, for one named case: a variable of the source that has, in
   the level cut from every use that is not bottom, an equation to the
   same constant, is that constant in the target, unless the target is
   bottom. (Simplify needs it to propagate a constant through a join
   point.) There must be such a use. *)
Definition join_keeps_constants {A : Type} (join : join_of_environments A)
    : Prop :=
  ∀ (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (x : variable) (c : const),
    join_inputs_well_formed avoid inputs ->
    uses_kinded inputs ->
    join avoid inputs = Some (target, analysis) ->
    knows (source inputs) x ->
    (∃ (index : Index.t) (use : Typing_env.t),
       use_at inputs index use ∧ Typing_env.is_bottom use = false) ->
    (∀ (index : Index.t) (use : Typing_env.t),
       use_at inputs index use ->
       Typing_env.is_bottom use = false ->
       ∃ (level : Typing_env_level.t) (ty : flambda_type) (alias : simple),
         Typing_env.cut use (cut_after inputs) = Some level
         ∧ Keyed.lookup name_eqb (Name_var x)
             (Typing_env_level.equations level)
           = Some ty
         ∧ get_alias_opt ty = Some alias
         ∧ Typing_env.get_canonical_simple_ignoring_name_mode use alias
           = Const c) ->
    Typing_env.is_bottom target = true
    ∨ Typing_env.get_canonical_simple_ignoring_name_mode target (Var x)
      = Const c.

(* The uses have the names of the source at the kinds at which the
   source has them. *)
Definition uses_agree_with_source_on_kinds {A : Type}
    (inputs : join_inputs A) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t) (n : name) (kind : flambda_kind)
    (ty : flambda_type),
    use_at inputs index use ->
    Typing_env.mem None use n = true ->
    Typing_env.find (source inputs) n (Some kind) = Some ty ->
    ∃ ty' : flambda_type, Typing_env.find use n (Some kind) = Some ty'.

(* Kinds are kept: if the source is kinded too, and the uses have the
   names of the source at the source's kinds, the target is kinded. So
   a meet, or another join, may follow. It is a statement of its own,
   as the one of precision is: [join_guarantees] does not need these
   two premises, and does not promise this. (In the meet both hold of
   the environments of the branches of a disjunction, which are made
   from the source by adding equations. In Simplify the uses are made
   so from the environment at the definition of the continuation, and
   the source from that one by defining the parameters, which no use
   knows.) *)
Definition join_keeps_kinded {A : Type} (join : join_of_environments A)
    : Prop :=
  ∀ (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A),
    join_inputs_well_formed avoid inputs ->
    uses_kinded inputs ->
    typing_env_kinded (source inputs) ->
    uses_agree_with_source_on_kinds inputs ->
    join avoid inputs = Some (target, analysis) ->
    typing_env_kinded target.

End Contract.

(* ======================================================================== *)
(* The join of types, inside a join of environments                         *)
(* ======================================================================== *)

(* This section is the interface between the two halves of the
   transcription. The functions that join types (the join functions of
   meet_and_n_way_join.ml, in FlambdaJoinTypes.v) thread the state of
   the join of environments ([Join_env.t], in FlambdaJoinEnv.v) and are
   given, for each type, the index of the environment in which it
   holds. They call four functions of the environment side:
   [Join_env.joined_env], [n_way_join_simples], [import_type] and
   [n_way_join_env_extension]; and the environment side calls the join
   of types, which it is given as an argument ([n_way_join_type]).

   Neither half sees the other's definitions. The state is any type
   with three readings: the environment of an index ([joined_env]), the
   definitions of the existential variables made so far, and the
   variables that exist in the target so far, each with its kind
   (those of the source, the imported ones and the existential ones:
   [Bindings_in_target_env.exists_in_target_env]); and with an
   invariant, [state_valid], which is the environment side's and of
   which the join of types knows only that the four functions keep it.
   The four contracts below are what each half proves of its functions
   and assumes of the other's. Each is in four parts: the invariant is
   kept; the upper bound, under any valuation of the target that
   extends that of the input; scope, that what is answered mentions
   only variables of the target; and kinds.

   Kinds. What is answered is of the kind of what was given
   (unconditionally for types; a [Simple] of a kind that is not that
   of values is not a symbol). And the answer is kinded in the target
   ([type_kinded_in_target]) at the states that are kinded
   ([state_kinded]): a second predicate of the environment side, which
   holds from the start of a join whose source is kinded and whose
   uses have the names of the source at the source's kinds, and of
   which the join of types knows only that the four functions keep it.
   It is what the target being kinded is proved from
   ([join_keeps_kinded]); the upper bounds do not need it.

   The upper bounds speak of every valuation that extends that of an
   input. That there is one is asked of the invariant
   ([valid_states_have_witnesses]): it is where it is shown that the
   variables that the join makes are new and that each has one
   definition. Without it a step could add definitions that nothing
   meets, and its upper bound would say nothing.

   Every function is given types, [Simple]s and extensions that mention
   only variables that the environment of their index knows
   ([join_args_in_scope]): the environment side gives the types that an
   input has for a name, and the join of types passes on their
   components and what [Expand_head] answers in that environment. Each
   index is given once ([join_args_distinct]). And the types given are
   kinded in the environment of their index ([type_kinded]), so that
   each [Simple] given to the join of [Simple]s is of the kind given:
   the components of a kinded type are kinded, by the definition of
   [aliases_with_kinds], and what [Expand_head] answers is kinded where
   the environment is ([joined_envs_kinded]). *)
Section Join_of_types.

Context (typing_env_well_formed : Typing_env.t -> Prop).

Context {state : Type}
  (joined_env : state -> Index.t -> option Typing_env.t)
  (definitions_of_existentials :
     state -> finite_map (Simples_in_joined_envs.t * flambda_kind))
  (kind_in_target_env : state -> variable -> option flambda_kind)
  (state_valid : state -> Prop)
  (state_kinded : state -> Prop).

(* A variable exists in the target so far. *)
Definition exists_in_target_env (s : state) (x : variable) : bool :=
  match kind_in_target_env s x with
  | Some _ => true
  | None => false
  end.

(* The environments joined are well formed. *)
Definition joined_envs_well_formed (s : state) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t),
    joined_env s index = Some use -> typing_env_well_formed use.

(* The environments joined are kinded. The join of types needs it of
   the types that it expands in them; the environment side, of the
   types that it looks up in them when it imports a variable or makes
   one for [Simple]s. It is a premise of each of the four contracts
   and not a part of [state_valid]: the environments joined do not
   change in a join ([state_grows]), so whoever has it at the first
   call has it at every later one. *)
Definition joined_envs_kinded (s : state) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t),
    joined_env s index = Some use -> typing_env_kinded use.

(* A valuation of the target extends a valuation of the input of index
   [index], as far as the state has fixed: it agrees on what that input
   knows, and the existential variables made so far have the values of
   their definitions there. *)
Definition extends_input (s : state) (index : Index.t)
    (valuation valuation' : env) : Prop :=
  ∃ use : Typing_env.t,
    joined_env s index = Some use
    ∧ agrees_on_known use valuation valuation'
    ∧ definitions_hold (definitions_of_existentials s) index valuation
        valuation'.

(* What the environment side must prove of its invariant: at a valid
   state, every valuation that satisfies an input extends to a
   valuation of the target. *)
Definition valid_states_have_witnesses : Prop :=
  ∀ (s : state) (index : Index.t) (use : Typing_env.t)
    (statics : static_env) (valuation : env),
    state_valid s ->
    joined_env s index = Some use ->
    typing_env_satisfied_strictly statics valuation use ->
    ∃ valuation' : env, extends_input s index valuation valuation'.

(* A step of the join keeps the inputs, the definitions made and the
   variables of the target, with their kinds. *)
Definition state_grows (s s' : state) : Prop :=
  (∀ index : Index.t, joined_env s' index = joined_env s index)
  ∧ (∀ (x : variable)
       (definition : Simples_in_joined_envs.t * flambda_kind),
       lookup x (definitions_of_existentials s) = Some definition ->
       lookup x (definitions_of_existentials s') = Some definition)
  ∧ ∀ (x : variable) (kind : flambda_kind),
      kind_in_target_env s x = Some kind ->
      kind_in_target_env s' x = Some kind.

(* What is given mentions only variables that an environment knows. *)
Definition simple_in_scope (tenv : Typing_env.t) (simple : simple) : Prop :=
  ∀ x : variable, x ∈ free_variables_of_simple simple -> knows tenv x.

Definition type_in_scope (tenv : Typing_env.t) (ty : flambda_type) : Prop :=
  ∀ x : variable, x ∈ free_variables ty -> knows tenv x.

Definition env_extension_in_scope (tenv : Typing_env.t)
    (extension : env_extension) : Prop :=
  ∀ x : variable,
    x ∈ free_variables_of_env_extension extension -> knows tenv x.

(* Each thing given is in the scope of the environment of its index. *)
Definition join_args_in_scope {X : Type}
    (in_scope : Typing_env.t -> X -> Prop) (s : state)
    (args : list (Index.t * X)) : Prop :=
  ∀ (index : Index.t) (x : X),
    In (index, x) args ->
    ∃ use : Typing_env.t, joined_env s index = Some use ∧ in_scope use x.

(* Each index is given once. The environment side gives the bindings of
   a map from indices; the join of types passes on, for a component, at
   most one thing for each element of a list that it was given, so its
   lists keep this. (The join of [Simple]s makes a map of its list:
   of two [Simple]s at one index it would keep one, and could not be an
   upper bound of both.) *)
Definition join_args_distinct {X : Type} (args : list (Index.t * X)) : Prop :=
  NoDup (map fst args).

(* What is answered mentions only variables of the target. *)
Definition simple_in_target (s : state) (simple : simple) : Prop :=
  ∀ x : variable,
    x ∈ free_variables_of_simple simple -> exists_in_target_env s x = true.

Definition type_in_target (s : state) (ty : flambda_type) : Prop :=
  ∀ x : variable, x ∈ free_variables ty -> exists_in_target_env s x = true.

Definition env_extension_in_target (s : state) (extension : env_extension)
    : Prop :=
  ∀ x : variable,
    x ∈ free_variables_of_env_extension extension ->
    exists_in_target_env s x = true.

(* A [Simple] is of a kind in the target so far, as [simple_of_kind]
   is for an environment. A symbol is a value, and is one that an
   environment joined knows. *)
(* CR mshinwell: 2026-11-12. The case of a symbol said only that the
   kind is that of values. That was not enough to add what a join
   answers to an environment: [simple_of_kind] of a symbol in an
   environment says also that the environment knows the symbol, and
   the meet's environment asks it of every alias of a type that it is
   given ([type_kinded], through [type_fits] of FlambdaMeetSpec.v).
   The symbols of the target are those of the source, and the
   premises of the two contracts for a join of environments say that
   a symbol that a use knows is known to the source
   ([symbols_of_uses_in_source] above; the fourth conjunct of
   [scoped_env_of], FlambdaMeetSpec.v, for the join that the meet
   calls). So "known to an environment joined" is "known to the
   target" there. The state has no reading for the source,
   which is why the clause is about the environments joined. A change
   to a reviewed definition, made overnight with the agreement of the
   owner of the join of types, whose statements about that join take
   this reading. *)
Definition simple_of_kind_in_target (s : state) (simple : simple)
    (kind : flambda_kind) : Prop :=
  match simple with
  | Const c => FlambdaTypeGrammar.kind (type_for_const c) = kind
  | Var x => kind_in_target_env s x = Some kind
  | Symbol sym =>
    kind = Value
    ∧ ∃ (index : Index.t) (use : Typing_env.t),
        joined_env s index = Some use
        ∧ Typing_env.mem None use (Name_symbol sym) = true
  end.

(* What is answered is kinded in the target so far, as [type_kinded]
   is for an environment. *)
Definition type_kinded_in_target (s : state) (ty : flambda_type) : Prop :=
  ∀ (simple : simple) (kind : flambda_kind),
    (simple, kind) ∈ aliases_with_kinds ty ->
    simple_of_kind_in_target s simple kind.

Definition env_extension_kinded (tenv : Typing_env.t)
    (extension : env_extension) : Prop :=
  ∀ (simple : simple) (kind : flambda_kind),
    (simple, kind) ∈ aliases_of_env_extension extension ->
    simple_of_kind tenv simple kind.

Definition env_extension_kinded_in_target (s : state)
    (extension : env_extension) : Prop :=
  ∀ (simple : simple) (kind : flambda_kind),
    (simple, kind) ∈ aliases_of_env_extension extension ->
    simple_of_kind_in_target s simple kind.

(* The join of types: [Meet_and_n_way_join.n_way_join]. If it answers a
   type, then at each input a value of that input's type, under a
   valuation that satisfies the input, has the type answered under any
   valuation of the target that extends it. Unknown says nothing. The
   type answered is of the kind of each type given (the compiler's
   function stops if they are not of one kind, and if there is none),
   and is kinded in the target at a state that is kinded. This is what
   the join of types is to be proved to meet, and what the join of
   environments assumes of the one it is given. *)
Definition n_way_join_type_sound
    (n_way_join :
       state -> list (Index.t * flambda_type)
       -> option (Or_unknown.t flambda_type * state)) : Prop :=
  ∀ (s s' : state) (ts : list (Index.t * flambda_type))
    (result : Or_unknown.t flambda_type),
    state_valid s ->
    joined_envs_well_formed s ->
    joined_envs_kinded s ->
    join_args_distinct ts ->
    join_args_in_scope type_in_scope s ts ->
    join_args_in_scope type_kinded s ts ->
    n_way_join s ts = Some (result, s') ->
    state_valid s'
    ∧ state_grows s s'
    ∧ (∀ ty : flambda_type,
         result = Or_unknown.Known ty ->
         type_in_target s' ty
         ∧ (∀ (index : Index.t) (t_index : flambda_type),
              In (index, t_index) ts -> kind ty = kind t_index)
         ∧ ∀ (index : Index.t) (t_index : flambda_type)
             (use : Typing_env.t) (statics : static_env)
             (valuation valuation' : env) (v : value),
             In (index, t_index) ts ->
             joined_env s' index = Some use ->
             typing_env_satisfied_strictly statics valuation use ->
             extends_input s' index valuation valuation' ->
             type_satisfied statics valuation v t_index ->
             type_satisfied statics valuation' v ty)
    ∧ (state_kinded s ->
       state_kinded s'
       ∧ ∀ ty : flambda_type,
           result = Or_unknown.Known ty -> type_kinded_in_target s' ty).

(* The join of [Simple]s: [Join_env.n_way_join_simples]. The [Simple]
   answered has, in the target, the value that each [Simple] given has
   in its input, where it has one. Bottom is answered only for no
   [Simple]. A symbol is answered only at the kind of values (it is
   then the canonical [Simple] of each one given, and those are of the
   kind given). At a state that is kinded, the [Simple] answered is of
   the kind given in the target. *)
(* CR mshinwell: each [Simple] given must be of the kind given, in the
   environment of its index. The compiler's function is not given
   this, and does not test it: it makes a variable of the kind for the
   [Simple]s, whatever their kinds (see [typing_env_kinded] for when it
   stops later and when it does not). The join of types gets the
   premise from the types that it is given being kinded
   ([n_way_join_type_sound]), and those from the use environments being
   kinded ([uses_kinded]). Decisions D10 and D12; discrepancy T5-9. *)
Definition n_way_join_simples_sound
    (n_way_join_simples :
       state -> flambda_kind -> list (Index.t * simple)
       -> option (Or_bottom.t simple * state)) : Prop :=
  ∀ (s s' : state) (kind : flambda_kind) (simples : list (Index.t * simple))
    (result : Or_bottom.t simple),
    state_valid s ->
    joined_envs_well_formed s ->
    joined_envs_kinded s ->
    join_args_distinct simples ->
    join_args_in_scope (fun use simple => simple_of_kind use simple kind) s
      simples ->
    n_way_join_simples s kind simples = Some (result, s') ->
    state_valid s'
    ∧ state_grows s s'
    ∧ (result = Or_bottom.Bottom -> simples = [])
    ∧ (∀ sym : symbol, result = Or_bottom.Ok (Symbol sym) -> kind = Value)
    ∧ (∀ joined : simple,
         result = Or_bottom.Ok joined ->
         simple_in_target s' joined
         ∧ ∀ (index : Index.t) (simple : simple) (use : Typing_env.t)
             (statics : static_env) (valuation valuation' : env)
             (v : value),
             In (index, simple) simples ->
             joined_env s' index = Some use ->
             typing_env_satisfied_strictly statics valuation use ->
             extends_input s' index valuation valuation' ->
             eval_simple valuation simple = Some v ->
             eval_simple valuation' joined = Some v)
    ∧ (state_kinded s ->
       state_kinded s'
       ∧ ∀ joined : simple,
           result = Or_bottom.Ok joined ->
           simple_of_kind_in_target s' joined kind).

(* [Join_env.import_type]: the variables of a type become variables of
   the target. The join of types calls it on a type that it puts in its
   answer as it stands. Nothing is said of values, and nothing need be:
   a type means the same under a valuation of an input whose
   environment knows its variables and under a valuation of the target
   that extends that one, since each of those variables has in the
   second what it has in the first, a value or none
   ([agrees_on_known]), and the meaning of a type depends only on what
   its variables have. At a state that is kinded, a type that is kinded
   in one of the inputs is kinded in the target. *)
Definition import_type_sound
    (import_type : state -> flambda_type -> option state) : Prop :=
  ∀ (s s' : state) (ty : flambda_type),
    state_valid s ->
    joined_envs_well_formed s ->
    joined_envs_kinded s ->
    import_type s ty = Some s' ->
    state_valid s'
    ∧ state_grows s s'
    ∧ type_in_target s' ty
    ∧ (state_kinded s ->
       (∃ (index : Index.t) (use : Typing_env.t),
          joined_env s index = Some use ∧ type_kinded use ty) ->
       state_kinded s' ∧ type_kinded_in_target s' ty).

(* The join of environment extensions:
   [Join_env.n_way_join_env_extension]. The extension answered holds, in
   the target, wherever one of the extensions given holds in its input.
   Bottom is answered only for no extension. At a state that is kinded,
   if each extension given is kinded in its input, the one answered is
   kinded in the target. (The compiler's function also takes the join
   of types and the meet, and uses neither.) *)
Definition n_way_join_env_extension_sound
    (n_way_join_env_extension :
       state -> list (Index.t * env_extension)
       -> option (Or_bottom.t (env_extension * state))) : Prop :=
  ∀ (s : state) (extensions : list (Index.t * env_extension))
    (result : Or_bottom.t (env_extension * state)),
    state_valid s ->
    joined_envs_well_formed s ->
    joined_envs_kinded s ->
    join_args_distinct extensions ->
    join_args_in_scope env_extension_in_scope s extensions ->
    n_way_join_env_extension s extensions = Some result ->
    (result = Or_bottom.Bottom -> extensions = [])
    ∧ (∀ (joined : env_extension) (s' : state),
         result = Or_bottom.Ok (joined, s') ->
         state_valid s'
         ∧ state_grows s s'
         ∧ env_extension_in_target s' joined
         ∧ ∀ (index : Index.t) (extension : env_extension)
             (use : Typing_env.t) (statics : static_env)
             (valuation valuation' : env),
             In (index, extension) extensions ->
             joined_env s' index = Some use ->
             typing_env_satisfied_strictly statics valuation use ->
             extends_input s' index valuation valuation' ->
             env_extension_satisfied statics valuation extension ->
             env_extension_satisfied statics valuation' joined)
    ∧ (state_kinded s ->
       join_args_in_scope env_extension_kinded s extensions ->
       ∀ (joined : env_extension) (s' : state),
         result = Or_bottom.Ok (joined, s') ->
         state_kinded s' ∧ env_extension_kinded_in_target s' joined).

(* The equations inside what is answered. A type has names with
   equations inside it (in the extensions of the cases of a row-like
   structure and of a variant), each at the kind of the type of its
   equation ([equation_names_with_kinds], FlambdaTypeGrammar.v). An
   environment in which a meet may be made has the names of the
   equations inside its types at those kinds ([types_equations_kinded],
   FlambdaTypingEnvKindedSpec.v), and the target of a join is to be
   such an environment. So, at a state that is kinded, what is answered
   is to have the names of its equations at their kinds in the target,
   if what is given has them at their kinds in the environments
   joined. These are three contracts of their own, beside the four
   above, and not a part of them: a join of types, an import and a
   join of extensions that are to be used for a target in which a meet
   is made are asked both. (The join of [Simple]s answers no
   equation.) *)

(* The names of the equations inside a type are of the kinds of their
   types in the target so far, as [type_equations_kinded] is for an
   environment. *)
Definition type_equations_kinded_in_target (s : state) (ty : flambda_type)
    : Prop :=
  ∀ (n : name) (kind : flambda_kind),
    (n, kind) ∈ equation_names_with_kinds ty ->
    simple_of_kind_in_target s (simple_of_name n) kind.

(* The same of an extension: the names of its equations, and of the
   equations inside their types. *)
Definition env_extension_equations_kinded (tenv : Typing_env.t)
    (extension : env_extension) : Prop :=
  ∀ (n : name) (kind : flambda_kind),
    (n, kind) ∈ equation_names_of_env_extension extension ->
    simple_of_kind tenv (simple_of_name n) kind.

Definition env_extension_equations_kinded_in_target (s : state)
    (extension : env_extension) : Prop :=
  ∀ (n : name) (kind : flambda_kind),
    (n, kind) ∈ equation_names_of_env_extension extension ->
    simple_of_kind_in_target s (simple_of_name n) kind.

(* The environments joined have the names of the equations inside
   their types at the kinds of those equations. The join of types
   needs it of the types that it expands in them. *)
Definition joined_envs_equations_kinded (s : state) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t),
    joined_env s index = Some use -> types_equations_kinded use.

(* The join of types, under the premises of [n_way_join_type_sound]:
   if the types given have the names of their equations at their
   kinds, each in the environment of its index, the type answered has
   them at their kinds in the target. *)
Definition n_way_join_type_equations_kinded
    (n_way_join :
       state -> list (Index.t * flambda_type)
       -> option (Or_unknown.t flambda_type * state)) : Prop :=
  ∀ (s s' : state) (ts : list (Index.t * flambda_type)) (ty : flambda_type),
    state_valid s ->
    joined_envs_well_formed s ->
    joined_envs_kinded s ->
    joined_envs_equations_kinded s ->
    join_args_distinct ts ->
    join_args_in_scope type_in_scope s ts ->
    join_args_in_scope type_kinded s ts ->
    join_args_in_scope type_equations_kinded s ts ->
    state_kinded s ->
    n_way_join s ts = Some (Or_unknown.Known ty, s') ->
    type_equations_kinded_in_target s' ty.

(* The import of a type: the type has in the target the names of its
   equations at the kinds at which an input has them. *)
Definition import_type_equations_kinded
    (import_type : state -> flambda_type -> option state) : Prop :=
  ∀ (s s' : state) (ty : flambda_type),
    state_valid s ->
    joined_envs_well_formed s ->
    joined_envs_kinded s ->
    state_kinded s ->
    (∃ (index : Index.t) (use : Typing_env.t),
       joined_env s index = Some use ∧ type_equations_kinded use ty) ->
    import_type s ty = Some s' ->
    type_equations_kinded_in_target s' ty.

(* The join of environment extensions: if each extension given has the
   names of its equations at their kinds in its input, the one
   answered has them at their kinds in the target. *)
Definition n_way_join_env_extension_equations_kinded
    (n_way_join_env_extension :
       state -> list (Index.t * env_extension)
       -> option (Or_bottom.t (env_extension * state))) : Prop :=
  ∀ (s s' : state) (extensions : list (Index.t * env_extension))
    (joined : env_extension),
    state_valid s ->
    joined_envs_well_formed s ->
    joined_envs_kinded s ->
    state_kinded s ->
    join_args_in_scope env_extension_equations_kinded s extensions ->
    n_way_join_env_extension s extensions = Some (Or_bottom.Ok (joined, s'))
    ->
    env_extension_equations_kinded_in_target s' joined.

End Join_of_types.

(* ======================================================================== *)
(* Statements that do not need the functions                                *)
(* ======================================================================== *)

(* The contract is not empty: the join that answers the source, and an
   analysis that says nothing, meets it. *)
Definition join_to_source_sound_statement : Prop :=
  ∀ (typing_env_well_formed : Typing_env.t -> Prop) (A : Type),
    sound_join_of_environments typing_env_well_formed
      (fun (_ : list variable) (inputs : join_inputs A) =>
         Some (source inputs, Analysis.empty)).

(* A join that meets the contract meets the one that is to be proved of
   the compiler's. *)
Definition sound_join_where_values_are_read_of_sound_statement : Prop :=
  ∀ (typing_env_well_formed : Typing_env.t -> Prop) (A : Type)
    (join : join_of_environments A),
    sound_join_of_environments typing_env_well_formed join ->
    sound_join_where_values_are_read typing_env_well_formed join.

(* A join that meets the contract for the runs that have the values
   that the join reads meets the stronger one, for the runs that have
   one more condition. The witness is the same: a variable that the
   join added an equation on has a value at the use by that condition
   ([added_equations_have_values]), as a variable that the use or the
   source knows, and then the witness has that value for it
   ([agrees_on_known]); or by its definition there, and then the
   witness has the value of the definition ([definitions_hold]). So the
   condition asks nothing more of the join: it is what the
   caller gives to be told that those variables have values. *)
Definition equations_have_values_of_values_are_read_statement : Prop :=
  ∀ (typing_env_well_formed : Typing_env.t -> Prop) (A : Type)
    (join : join_of_environments A),
    sound_join_where_values_are_read typing_env_well_formed join ->
    sound_join_where_equations_have_values typing_env_well_formed join.

(* And it is not met by everything: the join that answers a bottom
   environment does not meet it. *)
Definition join_to_bottom_unsound_statement : Prop :=
  ~ sound_join_of_environments (fun _ : Typing_env.t => True)
      (fun (_ : list variable) (inputs : join_inputs unit) =>
         Some (Typing_env.make_bottom (source inputs), Analysis.empty)).

(* In the machine. If the target admits a use, and every variable that
   may occur in terms in the target may occur in terms at the use, then
   a state of the machine that satisfies the use and the source
   satisfies the target; and the valuation that shows it gives each
   variable with a definition at that use, by a [Simple] that may occur
   in terms there, the value that the machine has for that [Simple].
   The last part is what entitles Simplify to add a parameter for such
   a variable and to pass that [Simple] for it at the use
   (join_points.ml, [add_extra_params_from_join_analysis]). *)
Definition join_admits_use_in_machine_statement : Prop :=
  ∀ (source use target : Typing_env.t)
    (definitions definitions_at_normal_mode
       : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (statics : static_env) (vars : env),
    join_admits_use source use target definitions definitions_at_normal_mode
      index ->
    (∀ x : variable, in_scope_in_terms target x -> in_scope_in_terms use x) ->
    use_and_source_satisfied source use statics vars ->
    ∃ valuation' : env,
      typing_env_satisfied_strictly statics valuation' target
      ∧ agrees_in_terms target valuation' vars
      ∧ bound_in_terms target vars
      ∧ ∀ (x : variable) (simples : Simples_in_joined_envs.t)
          (kind : flambda_kind) (simple : simple),
          lookup x definitions_at_normal_mode = Some (simples, kind) ->
          Simples_in_joined_envs.find index simples = Some simple ->
          Typing_env.mem_simple (Some Name_mode.normal) use simple = true ->
          lookup x valuation' = eval_simple vars simple.

Definition join_admits_use_env_satisfies_statement : Prop :=
  ∀ (source use target : Typing_env.t)
    (definitions definitions_at_normal_mode
       : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (statics : static_env) (vars : env),
    join_admits_use source use target definitions definitions_at_normal_mode
      index ->
    (∀ x : variable, in_scope_in_terms target x -> in_scope_in_terms use x) ->
    use_and_source_satisfied source use statics vars ->
    env_satisfies_strictly target statics vars.

(* The same of a join that meets the contract: if a variable that may
   occur in terms in the source may occur in terms at each use, a state
   of the machine that satisfies a use and the source satisfies the
   target. *)
Definition join_guarantees_in_machine_statement : Prop :=
  ∀ (typing_env_well_formed : Typing_env.t -> Prop) (A : Type)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (index : Index.t) (use : Typing_env.t)
    (statics : static_env) (vars : env),
    join_guarantees typing_env_well_formed avoid inputs target analysis ->
    terms_in_scope_at_uses inputs ->
    use_at inputs index use ->
    use_and_source_satisfied (source inputs) use statics vars ->
    env_satisfies_strictly target statics vars.

(* What the analysis says of a value at a use: where
   [Analysis.value_at_use] answers a constant for the [Simple] that a
   variable stands for at a use, the variable holds that constant under
   the valuation of the target for that use. Unknown says nothing, and
   so does [Not_refined_at_join]. *)
Definition known_value_at_use_statement : Prop :=
  ∀ (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (valuation valuation' : env) (x : variable)
    (simples : Simples_in_joined_envs.t) (kind : flambda_kind)
    (simple : simple) (c : const),
    definitions_hold definitions index valuation valuation' ->
    lookup x definitions = Some (simples, kind) ->
    Simples_in_joined_envs.find index simples = Some simple ->
    Analysis.value_at_use simple = Or_unknown.Known c ->
    lookup x valuation' = Some (Value_const c).

(* What the two lists that Simplify folds over hold
   ([Analysis.values_at_uses], [Analysis.definitions_at_uses]): each
   element is the identifier of an index and what the vector has at
   that index, so what is said above of a definition at an index is
   true of each element. *)
Definition values_at_uses_found_statement : Prop :=
  ∀ (A : Type) (external_ids : Keyed.finite_map Index.t A)
    (simples : Simples_in_joined_envs.t)
    (values : list (A * Or_unknown.t const)) (id : A)
    (value : Or_unknown.t const),
    Analysis.values_at_uses external_ids simples = Some values ->
    In (id, value) values ->
    ∃ (index : Index.t) (simple : simple),
      Keyed.lookup Index.equal index external_ids = Some id
      ∧ Simples_in_joined_envs.find index simples = Some simple
      ∧ value = Analysis.value_at_use simple.

Definition definitions_at_uses_found_statement : Prop :=
  ∀ (A : Type) (external_ids : Keyed.finite_map Index.t A)
    (simples : Simples_in_joined_envs.t) (definitions : list (A * simple))
    (id : A) (simple : simple),
    Analysis.definitions_at_uses external_ids simples = Some definitions ->
    In (id, simple) definitions ->
    ∃ index : Index.t,
      Keyed.lookup Index.equal index external_ids = Some id
      ∧ Simples_in_joined_envs.find index simples = Some simple.

(* ======================================================================== *)
(* The n-way join of closures entries is not an upper bound                 *)
(* ======================================================================== *)

(* A closure type with one case, in which only the function types
   vary. *)
Definition closures_type_with (slot : function_slot)
    (index : row_like_index set_of_closures_contents unit)
    (closure_types value_slot_types : finite_map flambda_type)
    (function_types : finite_map (Or_unknown.t function_type))
    : flambda_type :=
  create_closures Alloc_mode.Heap
    (Row_like_for_closures
       [(slot,
         Row_like_case
           (Closures_entry function_types closure_types value_slot_types)
           index empty_env_extension)]).

(* There are two closure types that differ only in their function
   types, and a value that has the first, such that the value has not
   the type whose function types are those that the compiler's n-way
   join made of the two before it was changed. (The first says nothing
   of the code of the closure's own function, the second names a code,
   and the value's code is another.) So [n_way_join_closures_entry]
   did not meet [n_way_join_type_sound]. *)
Definition n_way_join_function_types_before_fix_unsound_statement : Prop :=
  ∃ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (index : row_like_index set_of_closures_contents unit)
    (closure_types value_slot_types : finite_map flambda_type)
    (function_types0 function_types1
       : finite_map (Or_unknown.t function_type)),
    type_satisfied statics vars v
      (closures_type_with slot index closure_types value_slot_types
         function_types0)
    ∧ ~ type_satisfied statics vars v
          (closures_type_with slot index closure_types value_slot_types
             (n_way_join_function_types_before_fix
                [(0, function_types0); (1, function_types1)])).

(* With the change, the function types joined say of the code of a
   function only what every entry says: a closures entry that holds
   with the function types of one of the entries joined holds with the
   function types joined. *)
Definition n_way_join_function_types_sound_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (entries : list (Index.t * finite_map (Or_unknown.t function_type)))
    (index : Index.t)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type),
    In (index, function_types) entries ->
    closures_entry_satisfied statics vars v slot functions value_slots
      (Closures_entry function_types closure_types value_slot_types) ->
    closures_entry_satisfied statics vars v slot functions value_slots
      (Closures_entry (n_way_join_function_types entries)
         closure_types value_slot_types).
