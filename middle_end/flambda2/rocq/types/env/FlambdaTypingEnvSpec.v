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


(* What it is for the values of a run to satisfy a typing environment,
   and what looking a name up in the environment says of its value.

   This file is to be reviewed: [typing_env_satisfied_but], the notions
   made with it ([typing_env_satisfied_strictly] and
   [typing_env_satisfied]; [env_satisfies_strictly] and [env_satisfies]
   for the machine), what they are made of, and the statements. Every
   later statement about types that is relative to an environment
   (meet, join, the provers, the rewrites of Simplify) takes one of
   them as a premise. So they have to be true of the environments that
   Simplify builds, or those statements say nothing of real programs;
   and strong enough that the type found for a name is a fact about its
   value. The environment itself is FlambdaTypingEnv.v, which is to be
   read against middle_end/flambda2/types/env/typing_env.ml; what a
   type means is FlambdaTypeMeaningSpec.v; the aliases are
   FlambdaEnvAliasesSpec.v. Examples, with environments of the machine
   that do and do not satisfy an environment, are
   FlambdaTypingEnvExamplesSpec.v.

   The compiler has no such definition to compare with. What an
   environment means there is what its users take it to mean, and those
   are what to read this against: [Typing_env.find] and
   [type_simple_in_term_exn] as Simplify calls them
   (simplify/simplify_simple.ml), [Expand_head.expand_head] and the
   provers; how Simplify extends the environment as it goes under
   binders (simplify/env/downwards_env.ml: [define_variable],
   [add_variable], [add_parameters], [enter_set_of_closures]); and how
   the environment of a handler is made from those of its uses
   (simplify/join_points.ml, types/join_levels_old.ml and
   types/env/join_env.ml).

   Valuations. An environment knows variables that have no value where
   the term runs:
   - those of name mode [In_types]: the variables that a join makes
     existential ([Meet_env.add_env_extension_from_level], and the
     n-way join, which defines them with
     [add_variable_definition ... In_types]), and those of an extension
     with extra variables. Such a variable stands for the value that a
     variable bound in the body of a [Let_cont] had at the use that
     jumped to the handler;
   - those of name mode [Phantom], which are for debugging information;
   - those that are out of scope. In the body of a function the
     environment is that of the place where the closure is made, with
     [min_binding_time] moved to the present
     ([Typing_env.closure_env]): its variables are still there, and
     the types of the value slots are aliases to them, but the machine
     runs the body with the parameters and the closure and nothing
     else. Such a variable stands for the value it had where the
     closure was made.
   So an environment is satisfied by a valuation: an environment of
   values, like the machine's, that may give a value to any variable
   that the typing environment knows ([typing_env_satisfied_but]). The
   machine satisfies a typing environment when it binds the variables
   in scope and some valuation that agrees with it on them satisfies
   the environment ([env_satisfies]). Which variables are in scope is
   the compiler's own test ([in_scope_in_terms]). Statements about
   types (meet, join, the provers) are to be made of valuations, as the
   types of an environment mention variables that are not in scope;
   statements about terms are to be made of the machine.
   FlambdaTypeMeaningSpec.v needs nothing else of the environment, and
   what it calls [vars] is the valuation.

   A valuation need not give a value to every variable that the
   environment knows. A variable that a join has made existential may
   stand for a variable that some uses of the continuation bind and
   others do not (the n-way join brings such a variable in with the
   join of its types at the uses that define it:
   types/env/join_env.ml); in a run that comes through one of the
   others it stands for nothing. The compiler also builds environments
   that are not bottom in which a name has the type bottom: the binary
   join works in one, in which the variables that it has not yet joined
   have that type (types/join_levels_old.ml, with
   [Meet_env.add_env_extension_maybe_bottom]). Of a name with no value
   nothing is asked. The alias type of such a name is the type of no value
   (FlambdaTypeMeaningSpec.v), and no name that has a value is an alias
   of it (FlambdaEnvAliasesSpec.v): so where a type mentions it, the
   type speaks of a case that is not the case in that run. A name that
   is reached through a value in hand, by an alias type that the value
   has, has a value. A statement about a name that is not so reached
   needs the premise that the name has a value: to add an equation on a
   name ([Meet_env.add_equation]), or to conclude from the type of a
   name being bottom that the program point is not reached. An
   environment in which a variable that exists only in types has the
   type bottom is satisfied, strictly, by the valuations that give the
   variable no value.

   What it is to satisfy an environment, and what was decided.
   - Names. Every name that has a type ([names_to_types]) and a value
     in the valuation: the value has the type
     ([names_have_their_types_but]). The type of a name is the one that
     this map has for it: the one that the last [replace_equation] gave
     it. The equations of the levels are the history of the environment
     and are not read. A value that has a type has the kind of the type
     (FlambdaTypeMeaningSpec.v).
   - Symbols. A symbol is a value by itself, a pointer, so it always has
     a value; its type says what the static environment binds it to. A
     symbol that is defined and has no equation ([defined_symbols]) has
     the type of the values that are not null, which a pointer is: so
     nothing is asked of it, not even that the static environment binds
     it.
   - Code. Nothing is asked. A function type says which code id a
     closure has (FlambdaTypeMeaningSpec.v); that the static environment
     binds the code id, and to what, is not part of an environment of
     types. (Simplify keeps the code in its own environment,
     [Downwards_env.all_code].)
   - Names that are exempt. [typing_env_satisfied_but] is given the
     names that are exempt: of the value of an exempt name only the
     kind of its type is asked. Two cases have names.
     [typing_env_satisfied_strictly]: no name is exempt. This is the
     one for the statements about types. With an exempt name what the
     compiler does with types is not right: if [n] has the type of the
     number 4 and holds poison, poison has the alias type of [n] and
     has not the type found for [n], so that to put the type of a name
     for an alias to it, as [Expand_head.expand_head] does, is wrong,
     and a meet that reaches [n] can answer bottom where there are
     values ([alias_has_type_found_statement] and
     FlambdaTypingEnvExamplesSpec.v).
     To be strictly satisfied is not to be free of poison. A name may
     hold poison where its type admits it: the unknown type, an alias
     of a name that holds it, the type of the values that are not null.
     By decision D26 as amended, the relations [Is_int] and [Get_tag]
     of a type are never true of a scrutinee that holds poison. So a
     value that has a type with one of those two has a scrutinee that
     is not poison, and a statement that reads one of them from the
     type of a name that has a value has no case of poison; of a name
     with no value the type says nothing, as above
     ([relation_holds], FlambdaTypeMeaningSpec.v). [Is_null] is not so: it
     relates poison of kind value to 0, since poison is not null, and a
     statement that reads it has the case of a scrutinee that holds
     poison.
     (Where no name holds poison the two notions are one,
     [satisfied_no_poison_strictly_statement]; that is a narrower case
     than to be strictly satisfied.)
     [typing_env_satisfied]: the names that hold poison are exempt,
     and those whose type is an [Is_int] or a [Get_tag] relation on a
     name that holds poison ([exempt_by_poison]; decision D26; there is
     a CR there). Typing facts are about values that are not poison: a
     name that is not exempt has its type
     ([find_not_poison_statement]), and a use of a fact has the other
     case to deal with. Nothing else follows from it. It is there for
     what unboxing does: the parameters that stand for the fields of an
     inactive case of a variant are given their types on every path,
     and hold poison; and where such a field is unboxed in turn, its
     [Is_int] and tag parameters are given the relations to it as their
     types, and hold 0. Until then the environments that Simplify
     builds are strictly satisfied. Inside a type nothing changes: the
     fields of a block that is not poison have their types as
     FlambdaTypeMeaningSpec.v says, poison or not.
     The exemption of poison is by the value that the valuation gives,
     for every name. It cannot be only for the names in scope: a
     parameter that holds poison may be captured by a closure, and is
     out of scope in its body; and a variable that exists only in types
     may stand for one that a term bound at a use of a continuation,
     where it held poison. The environment does not record which is
     which.
   - Aliases are exact, poison or not: names in one class of the
     aliases have the same value, or none of them has a value
     ([aliases_respected], FlambdaEnvAliasesSpec.v). The compiler reads
     canonical elements from the aliases, not from the alias types, so
     this is asked of the aliases themselves. That the two agree is for
     the invariant of an environment.
   - An environment that is bottom is satisfied by nothing.
   - The levels, the scopes, the binding times and
     [next_binding_time] are not read: they are what the functions
     that build an environment and cut it need, and are for the
     invariant of an environment. Name modes and [min_binding_time]
     are read only to say which variables are in scope.

   Is it true of what Simplify builds? Where Simplify defines a
   variable and gives it a type, the machine binds the variable to a
   value of that type; a handler runs under the variables of the place
   of its definition; the body of a function, and the variables that a
   join makes existential, are what valuations are for. Where it is not
   known to be true, and a statement that takes it as a premise says
   nothing:
   - Lifting. Simplify gives a variable the type "is this symbol" when
     it lifts the block or the closure that the variable is bound to,
     and gives the symbol its type, at a place where the term as it was
     has bound no such symbol: only the term that Simplify produces
     binds it, earlier. The values of a run of the first term do not
     satisfy that environment. This is for the workstream on lifting
     (S5), which has two runs to relate; the environment is satisfied
     by the values of the second.
   - Unboxing of variants: from there on only [typing_env_satisfied]
     holds, of which the statements about types say nothing. The
     compiler relies on no fact about a parameter that holds poison
     being used on a path that runs; nothing checks that.
   - Kinds. That the machine binds a variable to a value of the kind it
     is defined with is a premise wherever a variable is defined; it is
     for the semantics to give (workstream C2).
   - What FlambdaTypeMeaningSpec.v leaves out: code age (a type that
     names code), allocation modes.

   What is stated: how the notions compare; that satisfaction depends
   only on the values of the variables that the environment mentions,
   and for the machine only on those in scope; what satisfies the empty
   environment and a bottom one; what [find] and [mem] say of a name;
   what the canonical element of a simple is, with
   [type_simple_in_term_exn], which is what Simplify does to every
   simple of a term. No statement here has a premise on the environment
   (an invariant): they hold of every environment. What the functions
   that build an environment do to satisfaction, binders, the invariant
   and the renaming of names are to follow.

   The proofs are in FlambdaTypingEnvProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv.
Import ListNotations.

(* ======================================================================== *)
(* What satisfies a typing environment                                      *)
(* ======================================================================== *)

(* A valuation is an environment of values, like the machine's. It may
   give a value to any variable that a typing environment knows, also
   to one that has no value at run time; and it need not give one. *)

(* What is asked of the value of a name whose type is [ty]: it has the
   type; or the name is exempt, and then the value has the kind of the
   type and nothing more is asked of it. *)
Definition has_type_unless (exempt : Prop) (statics : static_env)
    (valuation : env) (v : value) (ty : flambda_type) : Prop :=
  (exempt ∧ kind_of_value v = kind ty)
  ∨ type_satisfied statics valuation v ty.

(* Every name that the typing environment has a type for, and that has
   a value in the valuation: the value has the type, unless the name is
   exempt. *)
Definition names_have_their_types_but (exempt : name -> Prop)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t) : Prop :=
  ∀ (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) (v : value),
    Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    value_of_name valuation n = Some v ->
    has_type_unless (exempt n) statics valuation v ty.

(* A valuation satisfies a typing environment, but for the names that
   are exempt: the environment is not bottom, its names have their
   types, and names that are aliases have the same value. *)
Definition typing_env_satisfied_but (exempt : name -> Prop)
    (statics : static_env) (valuation : env) (tenv : Typing_env.t) : Prop :=
  Typing_env.is_bottom tenv = false
  ∧ names_have_their_types_but exempt statics valuation tenv
  ∧ aliases_respected (Typing_env.aliases tenv) valuation.

(* A valuation satisfies a typing environment strictly: no name is
   exempt. The statements about types (meet, join, the provers) are to
   be made of this one. *)
Definition typing_env_satisfied_strictly (statics : static_env)
    (valuation : env) (tenv : Typing_env.t) : Prop :=
  typing_env_satisfied_but (fun _ => False) statics valuation tenv.

(* A name holds poison in a valuation. *)
Definition holds_poison (valuation : env) (n : name) : Prop :=
  ∃ v : value, value_of_name valuation n = Some v ∧ is_poison v.

(* A type is, at its head, an [Is_int] or a [Get_tag] relation on a
   name that holds poison. This is the type that unboxing gives the
   [Is_int] or the tag parameter of a field of an inactive case of a
   variant; the parameter holds 0, and neither relation is true of a
   scrutinee that holds poison. [Is_null] is not among them: it relates
   poison of kind value to 0 (FlambdaTypeMeaningSpec.v), so that
   relation on a name that holds poison is true at 0 and exempts no
   name. *)
Definition relation_on_poison (valuation : env) (ty : flambda_type) : Prop :=
  ∃ (head : head_of_kind_naked_immediate) (relation : Relation.t)
    (scrutinee : name),
    ty = Type_naked_immediate (Ok (No_alias head))
    ∧ (relation, scrutinee) ∈ inverse_relations head
    ∧ (relation = Relation.Is_int ∨ relation = Relation.Get_tag)
    ∧ holds_poison valuation scrutinee.

(* The names that are exempt where poison is: a name that holds poison,
   and a name whose type is an [Is_int] or a [Get_tag] relation on a
   name that holds poison. *)
Definition exempt_by_poison (valuation : env) (tenv : Typing_env.t)
    (n : name) : Prop :=
  holds_poison valuation n
  ∨ ∃ (ty : flambda_type)
      (binding_time_and_mode : Binding_time.With_name_mode.t),
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
      = Some (ty, binding_time_and_mode)
      ∧ relation_on_poison valuation ty.

(* A valuation satisfies a typing environment: the names that are
   exempt where poison is are exempt. *)
(* CR mshinwell: nothing is asked of the type of a name whose value is
   poison. That is a decision of the model, which the compiler does not
   state. Unboxing gives the parameters that stand for the fields of an
   inactive case of a variant their types on every path, and passes
   poison for them (simplify/unboxing/unboxing_epa.ml,
   build_unboxing_denv.ml). It relies on there being no strict use of
   such a parameter where it holds poison, and no meet of its type to
   bottom there; nothing checks either. The same of the [Is_int] and
   tag parameters of such a field that is unboxed in turn, for which
   unboxing passes 0. Decision D26. *)
(* CR mshinwell: a valuation may give poison to a variable that exists
   only in types, of which nothing is then asked: the exemption of
   poison is by the value, for every name, and not only for the names
   that a term binds. The alternative is to exempt only the variables
   in scope, those that a closure has captured and those that a join
   has made existential from a variable of a use; the environment does
   not record the last two. Nothing is known to go wrong with the
   choice made. Decision D26. *)
Definition typing_env_satisfied (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) : Prop :=
  typing_env_satisfied_but (exempt_by_poison valuation tenv) statics
    valuation tenv.

(* A variable may occur in a term simplified under the typing
   environment: it is bound there, with the normal name mode, and is not
   out of scope. This is the compiler's
   [Typing_env.mem ~min_name_mode:Name_mode.normal]. *)
Definition in_scope_in_terms (tenv : Typing_env.t) (x : variable) : Prop :=
  Typing_env.mem (Some Name_mode.normal) tenv (Name_var x) = true.

(* The machine binds the variables that may occur in terms. *)
Definition bound_in_terms (tenv : Typing_env.t) (vars : env) : Prop :=
  ∀ x : variable,
    in_scope_in_terms tenv x -> ∃ v : value, lookup x vars = Some v.

(* A valuation gives the variables that may occur in terms the values
   that the machine has for them. *)
Definition agrees_in_terms (tenv : Typing_env.t) (valuation vars : env)
    : Prop :=
  ∀ x : variable,
    in_scope_in_terms tenv x -> lookup x valuation = lookup x vars.

(* The static environment and the variables of a state of the machine
   satisfy a typing environment: the machine binds the variables in
   scope, and some valuation satisfies the environment that gives them
   the values that the machine has for them. The other variables of the
   typing environment (those of name mode [In_types] or [Phantom], and
   those that are out of scope) have the values that the valuation
   chooses, or none, whatever the machine has. *)
Definition env_satisfies (tenv : Typing_env.t) (statics : static_env)
    (vars : env) : Prop :=
  bound_in_terms tenv vars
  ∧ ∃ valuation : env,
      typing_env_satisfied statics valuation tenv
      ∧ agrees_in_terms tenv valuation vars.

(* The same with no name exempt. *)
Definition env_satisfies_strictly (tenv : Typing_env.t)
    (statics : static_env) (vars : env) : Prop :=
  bound_in_terms tenv vars
  ∧ ∃ valuation : env,
      typing_env_satisfied_strictly statics valuation tenv
      ∧ agrees_in_terms tenv valuation vars.

(* What is asked of a value fixes its kind, exempt or not; and the
   value of a name that is not exempt has the type. *)
Definition has_type_unless_kind_statement : Prop :=
  ∀ (exempt : Prop) (statics : static_env) (valuation : env) (v : value)
    (ty : flambda_type),
    has_type_unless exempt statics valuation v ty ->
    kind_of_value v = kind ty.

Definition has_type_unless_not_exempt_statement : Prop :=
  ∀ (exempt : Prop) (statics : static_env) (valuation : env) (v : value)
    (ty : flambda_type),
    has_type_unless exempt statics valuation v ty ->
    ~ exempt ->
    type_satisfied statics valuation v ty.

(* ======================================================================== *)
(* How the notions compare                                                  *)
(* ======================================================================== *)

(* The more names are exempt, the less is asked. *)
Definition satisfied_but_weaken_statement : Prop :=
  ∀ (exempt exempt' : name -> Prop) (statics : static_env)
    (valuation : env) (tenv : Typing_env.t),
    (∀ n : name, exempt n -> exempt' n) ->
    typing_env_satisfied_but exempt statics valuation tenv ->
    typing_env_satisfied_but exempt' statics valuation tenv.

(* So a valuation that satisfies an environment strictly satisfies
   it... *)
Definition satisfied_strictly_satisfied_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t),
    typing_env_satisfied_strictly statics valuation tenv ->
    typing_env_satisfied statics valuation tenv.

(* ... and the two are the same of a valuation in which no name holds
   poison. *)
Definition satisfied_no_poison_strictly_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t),
    (∀ n : name, ~ holds_poison valuation n) ->
    typing_env_satisfied statics valuation tenv ->
    typing_env_satisfied_strictly statics valuation tenv.

Definition env_satisfies_strictly_env_satisfies_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env),
    env_satisfies_strictly tenv statics vars ->
    env_satisfies tenv statics vars.

(* ======================================================================== *)
(* What satisfaction depends on                                             *)
(* ======================================================================== *)

(* A typing environment mentions a variable: it has a type for it, a
   type that it has for a name mentions it, or its aliases do. (In an
   environment that Simplify builds these are bound in it: that is for
   the invariant.) *)
Definition typing_env_mentions (tenv : Typing_env.t) (x : variable) : Prop :=
  (∃ (ty : flambda_type)
     (binding_time_and_mode : Binding_time.With_name_mode.t),
     Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
     = Some (ty, binding_time_and_mode))
  ∨ (∃ (n : name) (ty : flambda_type)
       (binding_time_and_mode : Binding_time.With_name_mode.t),
       Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
       = Some (ty, binding_time_and_mode)
       ∧ x ∈ free_variables ty)
  ∨ aliases_mention (Typing_env.aliases tenv) (Name_var x).

(* Whether a valuation satisfies an environment depends only on the
   values of the variables that the environment mentions. So binding a
   variable that it does not mention, or leaving it out, changes
   nothing. *)
Definition satisfied_but_coincide_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env)
    (valuation valuation' : env) (tenv : Typing_env.t),
    (∀ x : variable,
      typing_env_mentions tenv x ->
      lookup x valuation = lookup x valuation') ->
    typing_env_satisfied_but exempt statics valuation tenv ->
    typing_env_satisfied_but exempt statics valuation' tenv.

Definition typing_env_satisfied_coincide_statement : Prop :=
  ∀ (statics : static_env) (valuation valuation' : env)
    (tenv : Typing_env.t),
    (∀ x : variable,
      typing_env_mentions tenv x ->
      lookup x valuation = lookup x valuation') ->
    typing_env_satisfied statics valuation tenv ->
    typing_env_satisfied statics valuation' tenv.

(* Whether the machine satisfies an environment depends only on the
   values of the variables in scope. *)
Definition env_satisfies_coincide_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars vars' : env),
    (∀ x : variable,
      in_scope_in_terms tenv x -> lookup x vars = lookup x vars') ->
    env_satisfies tenv statics vars ->
    env_satisfies tenv statics vars'.

Definition env_satisfies_strictly_coincide_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars vars' : env),
    (∀ x : variable,
      in_scope_in_terms tenv x -> lookup x vars = lookup x vars') ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly tenv statics vars'.

(* The variables of the machine that give every name its type satisfy
   the environment: no other valuation need be found. *)
Definition satisfied_strictly_env_satisfies_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env),
    bound_in_terms tenv vars ->
    typing_env_satisfied_strictly statics vars tenv ->
    env_satisfies_strictly tenv statics vars.

Definition typing_env_satisfied_env_satisfies_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env),
    bound_in_terms tenv vars ->
    typing_env_satisfied statics vars tenv ->
    env_satisfies tenv statics vars.

(* ======================================================================== *)
(* What every environment satisfies, and what none does                     *)
(* ======================================================================== *)

(* The environment that knows nothing is satisfied by everything. *)
Definition create_satisfied_statement : Prop :=
  ∀ (statics : static_env) (vars : env),
    env_satisfies_strictly Typing_env.create statics vars.

(* An environment that is bottom is satisfied by nothing, whatever is
   exempt: a program point whose environment is bottom is not
   reached. *)
Definition bottom_not_satisfied_but_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t),
    Typing_env.is_bottom tenv = true ->
    ~ typing_env_satisfied_but exempt statics valuation tenv.

Definition bottom_not_satisfied_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env),
    Typing_env.is_bottom tenv = true -> ~ env_satisfies tenv statics vars.

Definition make_bottom_not_satisfied_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env),
    ~ env_satisfies (Typing_env.make_bottom tenv) statics vars.

(* ======================================================================== *)
(* Looking a name up                                                        *)
(* ======================================================================== *)

(* The type that the environment gives a name is the type of its value,
   if it has one and is not exempt, under a valuation that satisfies
   the environment. *)
Definition find_satisfied_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (n : name) (kind : option flambda_kind)
    (ty : flambda_type) (v : value),
    typing_env_satisfied_but exempt statics valuation tenv ->
    Typing_env.find tenv n kind = Some ty ->
    value_of_name valuation n = Some v ->
    has_type_unless (exempt n) statics valuation v ty.

(* Strictly: the value of a name has the type found for it. *)
Definition find_satisfied_strictly_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (n : name) (kind : option flambda_kind) (ty : flambda_type) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    Typing_env.find tenv n kind = Some ty ->
    value_of_name valuation n = Some v ->
    type_satisfied statics valuation v ty.

(* So, strictly, a value of the alias type of a name has the type found
   for the name: an alias may be replaced by the type of what it is an
   alias of. *)
Definition alias_has_type_found_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (n : name) (kind : option flambda_kind) (ty : flambda_type)
    (k : flambda_kind) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    Typing_env.find tenv n kind = Some ty ->
    type_satisfied statics valuation v (alias_type_of k (simple_of_name n)) ->
    type_satisfied statics valuation v ty.

(* Where poison is exempt: a name whose value is not poison has the
   type found for it, unless that type is a relation on a name that
   holds poison. *)
Definition find_not_poison_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (n : name) (kind : option flambda_kind) (ty : flambda_type) (v : value),
    typing_env_satisfied statics valuation tenv ->
    Typing_env.find tenv n kind = Some ty ->
    value_of_name valuation n = Some v ->
    ~ is_poison v ->
    ~ relation_on_poison valuation ty ->
    type_satisfied statics valuation v ty.

(* The type found is of the kind asked for. *)
Definition find_kind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (kind : flambda_kind)
    (ty : flambda_type),
    Typing_env.find tenv n (Some kind) = Some ty ->
    FlambdaTypeGrammar.kind ty = kind.

(* What is found, whether the environment is bottom or not: the type
   that the environment has for the name; a bottom type, in an
   environment that is bottom; or, for a symbol that is defined and has
   no type, the type of the values that are not null. *)
Definition find_cases_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (kind : option flambda_kind)
    (ty : flambda_type),
    Typing_env.find tenv n kind = Some ty ->
    (∃ binding_time_and_mode : Binding_time.With_name_mode.t,
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
      = Some (ty, binding_time_and_mode))
    ∨ (∃ k : flambda_kind, ty = bottom k)
    ∨ ty = any_non_null_value.

(* A name that is bound has a type. *)
Definition mem_find_statement : Prop :=
  ∀ (tenv : Typing_env.t) (min_name_mode : option Name_mode.t) (n : name),
    Typing_env.mem min_name_mode tenv n = true ->
    ∃ ty : flambda_type, Typing_env.find tenv n None = Some ty.

(* In the machine, a variable in scope has a value, of the kind of its
   type. *)
Definition in_scope_in_terms_bound_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env)
    (x : variable) (ty : flambda_type),
    env_satisfies tenv statics vars ->
    in_scope_in_terms tenv x ->
    Typing_env.find tenv (Name_var x) None = Some ty ->
    ∃ v : value,
      lookup x vars = Some v
      ∧ kind_of_value v = FlambdaTypeGrammar.kind ty.

(* The type of a variable in scope is the type of the value that the
   machine has for it, if the type mentions only variables in scope:
   strictly; and where poison is exempt, unless that value is poison or
   the type is a relation on a variable that holds poison. (A type that
   mentions other variables is about the values that the valuation
   gives them: [find_satisfied_statement].) *)
Definition find_variable_in_terms_strictly_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env)
    (x : variable) (kind : option flambda_kind) (ty : flambda_type)
    (v : value),
    env_satisfies_strictly tenv statics vars ->
    in_scope_in_terms tenv x ->
    Typing_env.find tenv (Name_var x) kind = Some ty ->
    (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y) ->
    lookup x vars = Some v ->
    type_satisfied statics vars v ty.

Definition find_variable_in_terms_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env)
    (x : variable) (kind : option flambda_kind) (ty : flambda_type)
    (v : value),
    env_satisfies tenv statics vars ->
    in_scope_in_terms tenv x ->
    Typing_env.find tenv (Name_var x) kind = Some ty ->
    (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y) ->
    lookup x vars = Some v ->
    ~ is_poison v ->
    ~ relation_on_poison vars ty ->
    type_satisfied statics vars v ty.

(* The type of a symbol is the type of the pointer that the symbol is,
   which is not poison: it says what the static environment binds the
   symbol to. *)
Definition find_symbol_in_terms_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (s : symbol)
    (kind : option flambda_kind) (ty : flambda_type),
    env_satisfies tenv statics vars ->
    Typing_env.find tenv (Name_symbol s) kind = Some ty ->
    (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y) ->
    type_satisfied statics vars (Value_symbol s) ty.

(* ======================================================================== *)
(* Canonical elements                                                       *)
(* ======================================================================== *)

(* These ask only that the aliases are respected, so they hold whatever
   is exempt. *)

(* The canonical element of a [Simple] has the value of the [Simple],
   or neither has a value. *)
Definition canonical_ignoring_name_mode_satisfied_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (s : simple),
    typing_env_satisfied_but exempt statics valuation tenv ->
    eval_simple valuation
      (Typing_env.get_canonical_simple_ignoring_name_mode tenv s)
    = eval_simple valuation s.

Definition get_canonical_simple_satisfied_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t)
    (min_name_mode name_mode_of_existing_simple : option Name_mode.t)
    (s canonical : simple),
    typing_env_satisfied_but exempt statics valuation tenv ->
    Typing_env.get_canonical_simple_exn tenv min_name_mode
      name_mode_of_existing_simple s
    = Some (Some canonical) ->
    eval_simple valuation canonical = eval_simple valuation s.

(* The canonical element of the alias that a type is: a value of the
   type is the value of that element. *)
Definition get_alias_then_canonical_simple_satisfied_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t)
    (min_name_mode name_mode_of_existing_simple : option Name_mode.t)
    (typ : flambda_type) (canonical : simple) (v : value),
    typing_env_satisfied_but exempt statics valuation tenv ->
    Typing_env.get_alias_then_canonical_simple_exn tenv min_name_mode
      name_mode_of_existing_simple typ
    = Some (Some canonical) ->
    type_satisfied statics valuation v typ ->
    eval_simple valuation canonical = Some v.

(* The type of a [Simple] that occurs in a term: the canonical element
   answered has the value of the [Simple], and that value, if there is
   one, has the type answered. *)
Definition type_simple_in_term_satisfied_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (min_name_mode : option Name_mode.t)
    (s canonical : simple) (ty : flambda_type),
    typing_env_satisfied_but exempt statics valuation tenv ->
    Typing_env.type_simple_in_term_exn tenv min_name_mode s
    = Some (Some (ty, canonical)) ->
    eval_simple valuation canonical = eval_simple valuation s
    ∧ ∀ v : value,
        eval_simple valuation s = Some v ->
        type_satisfied statics valuation v ty.

(* In the machine: a [Simple] of a term may be replaced by the canonical
   element that the environment answers for the normal name mode. This
   is what Simplify does to every [Simple]
   ([Simplify_simple.simplify_simple]). *)
Definition type_simple_in_term_canonical_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env)
    (s canonical : simple) (ty : flambda_type),
    env_satisfies tenv statics vars ->
    Typing_env.mem_simple (Some Name_mode.normal) tenv s = true ->
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s
    = Some (Some (ty, canonical)) ->
    eval_simple vars canonical = eval_simple vars s.
