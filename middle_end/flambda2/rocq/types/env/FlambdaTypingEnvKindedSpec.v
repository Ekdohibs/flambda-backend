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


(* The kinds of the [Simple]s that a typing environment knows, and of
   those that it knows to be equal.

   This file is to be reviewed: the definitions and the statements. The
   environment is FlambdaTypingEnv.v; its invariant is
   FlambdaTypingEnvInvariantSpec.v.

   What is defined here is not part of the invariant of an environment,
   and is not to be. It is about agreement of kinds, and on kinds the
   model asks no more than the compiler checks whatever its flags
   (decisions D10, D12 and D32): programs that the compiler accepts
   break agreement of kinds, and Simplify must take their terms. So the
   conditions here are premises of the statements that need them (those
   of [Expand_head], of meet and of the join), written once here so
   that those statements share one text, with what each function of the
   environment does to them.

   - [simple_of_kind tenv simple kind]: the environment has the
     [Simple] at the kind. This is what the compiler asks where it
     follows an alias or is about to give a name a type:
     [Typing_env.find] with a kind, a fatal error whatever the flags if
     the type found is of another kind.
   - [aliases_kinded tenv]: two [Simple]s that the environment knows to
     be equal are of one kind. The compiler does not check it:
     [Typing_env.add_alias] and [Aliases.add] do not look at kinds. A
     meet makes two [Simple]s equal only where it has two types of one
     kind in hand ([Meet_env.add_equation_on_simple] passes the kind of
     the type to [record_demotion], which reads the type of the name
     demoted with it), so the environments that a meet builds from one
     that has the condition have it, if every alias inside the types
     that it meets is at the kind of its target, which is the next
     condition.
   - [type_kinded tenv ty] and [types_kinded tenv]: every alias inside
     the type, and inside every type that the environment has for a
     name, is at the kind that the environment has for its target
     ([aliases_with_kinds], FlambdaTypeGrammar.v, lists them). The
     compiler finds out only where it follows the alias.
   - [symbols_of_kind_value tenv]: a symbol that has a type has one of
     the kind of values. A symbol is a value; one that is defined and
     has no type is found at the type of the values that are not null.
   - [typing_env_kinded tenv]: the last three together, the one premise
     about kinds that the statements about meet and the join take.
   - [type_equations_kinded tenv ty] and [types_equations_kinded
     tenv]: every name that has an equation in an extension inside the
     type, and inside every type that the environment has for a name,
     is in the environment of the kind of the type of that equation
     ([equation_names_with_kinds], FlambdaTypeGrammar.v, lists them).
     The compiler finds out where it adds the extension to an
     environment: a meet reads the type of a name with the kind of the
     type that it is about to give it. This one is for the meet, which
     adds the extensions of the types that it meets. It is beside
     [typing_env_kinded] and not a part of it: the statements about
     the join are made of [typing_env_kinded] as it is, and do not say
     that a join keeps this one.
   Nothing here asks anything of the names of which a naked immediate
   is the result of a relation ([inverse_relations]), which are of the
   kind of values and which neither list has; no statement has needed
   it.
   - [constants_of_kind tenv]: where the canonical element of a
     [Simple] is a constant, the constant is of the kind of the
     [Simple]. [Expand_head.expand_head] answers the type of the
     constant without looking at the kind of the alias, so this is what
     makes the type expanded be of the kind of the type. It follows
     from [aliases_kinded]
     ([aliases_kinded_constants_of_kind_statement]); alone, it is not
     kept when two [Simple]s are made equal.

   The statements: the kind of a name read on the map of the
   environment; that a [Simple] has one kind at most; what each
   function that builds an environment does to [simple_of_kind]; that
   each keeps [aliases_kinded], and [typing_env_kinded], under what it
   needs; what a kinded environment gives: the kind of a symbol, of
   a canonical element, of a constant that is one, and that a type
   found is kinded; and of the equations inside types, that they stay
   kinded where the [Simple]s keep their kinds, that those of a type
   found are kinded, and that each function keeps them kinded.

   The proofs are in FlambdaTypingEnvKindedProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvInvariantSpec.
Import ListNotations.

(* ======================================================================== *)
(* The conditions                                                           *)
(* ======================================================================== *)

(* A [Simple] is of a kind in an environment: the constant is of that
   kind, or the environment has a type of that kind for the name. *)
Definition simple_of_kind (tenv : Typing_env.t) (simple : simple)
    (kind : flambda_kind) : Prop :=
  match simple with
  | Const c => FlambdaTypeGrammar.kind (type_for_const c) = kind
  | Var x =>
    ∃ ty : flambda_type,
      Typing_env.find tenv (Name_var x) (Some kind) = Some ty
  | Symbol s =>
    ∃ ty : flambda_type,
      Typing_env.find tenv (Name_symbol s) (Some kind) = Some ty
  end.

(* Two [Simple]s that the environment knows to be equal are of one
   kind, if each is of a kind. *)
(* CR mshinwell: this is a premise about kinds, and not part of the
   invariant of an environment: on kinds the model asks no more than
   the compiler checks (decisions D10, D12 and D32). The compiler does
   not check it: [Typing_env.add_alias] and [Aliases.add] do not look
   at kinds. Where a meet makes two [Simple]s equal it has their types
   in hand, of one kind, and reads the type of the name demoted with
   that kind ([Meet_env.record_demotion], [Typing_env.find] with a
   kind), which is a fatal error if it is of another. Whether it ought
   to be an invariant, and whether ill-kinded code that the compiler
   accepts can make a name of one kind an alias of a [Simple] of
   another, is not known. *)
Definition aliases_kinded (tenv : Typing_env.t) : Prop :=
  ∀ (e e' : simple) (kind kind' : flambda_kind),
    aliased (Typing_env.aliases tenv) e e' ->
    simple_of_kind tenv e kind ->
    simple_of_kind tenv e' kind' ->
    kind = kind'.

(* Where the canonical element of a [Simple] of some kind is a
   constant, the constant is of that kind. The canonical element is the
   one that [Expand_head.expand_head] asks for, of a name mode that may
   be that of the variables that exist only in types. *)
Definition constants_of_kind (tenv : Typing_env.t) : Prop :=
  ∀ (s : simple) (kind : flambda_kind) (c : const),
    simple_of_kind tenv s kind ->
    Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types) None s
    = Some (Some (Const c)) ->
    FlambdaTypeGrammar.kind (type_for_const c) = kind.

(* A type is kinded in an environment: each of its aliases is at the
   kind that the environment has for its target, and its relation
   variables are naked immediates there. *)
Definition type_kinded (tenv : Typing_env.t) (ty : flambda_type) : Prop :=
  ∀ (s : simple) (kind : flambda_kind),
    (s, kind) ∈ aliases_with_kinds ty -> simple_of_kind tenv s kind.

(* Every type that the environment has for a name is kinded. *)
(* CR mshinwell: this is a premise of the statements that need it; it
   is not claimed that the compiler's environments have it. A type does
   not say at which kind the target of an alias is bound, the
   well-formedness of types leaves it out on purpose, and ill-kinded
   code, which Simplify must accept (decisions D10 and D12), may give a
   name an alias at another kind. The compiler finds out where it
   follows the alias ([Typing_env.find] with the kind of the alias, a
   fatal error whatever the flags), and not where the depth limit of a
   join is reached first. Whether an environment that is not kinded
   can reach a meet or a join is not known. Discrepancy T5-9. *)
Definition types_kinded (tenv : Typing_env.t) : Prop :=
  ∀ (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    type_kinded tenv ty.

(* A symbol that has a type has one of the kind of values. *)
(* The compiler does not check this where a symbol is given a type
   ([Typing_env.replace_equation]). A symbol that is defined and has no
   type is found at the type of the values that are not null, and the
   callers in [Meet_env] read the type of a name with the kind of the
   type that they are about to give it, so that from an environment
   that has the condition they make one that has it. *)
Definition symbols_of_kind_value (tenv : Typing_env.t) : Prop :=
  ∀ (s : symbol) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb (Name_symbol s) (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    FlambdaTypeGrammar.kind ty = Value.

(* An environment is kinded. *)
Definition typing_env_kinded (tenv : Typing_env.t) : Prop :=
  types_kinded tenv ∧ aliases_kinded tenv ∧ symbols_of_kind_value tenv.

(* The equations inside a type are kinded in an environment: each name
   that has an equation in an extension inside the type is, in the
   environment, of the kind of the type of that equation. *)
Definition type_equations_kinded (tenv : Typing_env.t) (ty : flambda_type)
    : Prop :=
  ∀ (n : name) (kind : flambda_kind),
    (n, kind) ∈ equation_names_with_kinds ty ->
    simple_of_kind tenv (simple_of_name n) kind.

(* The same of every type that the environment has for a name. *)
(* As for [types_kinded], this is a premise of the statements that need
   it, and it is not claimed that the compiler's environments have it:
   nothing checks the equations of an extension where a type that
   holds it is made or given to a name. The compiler finds out where
   it adds the extension to an environment ([Meet_env.add_equation]
   reads the type of the name with the kind of the type of the
   equation, [Typing_env.find] with a kind, a fatal error whatever the
   flags if the two differ). *)
Definition types_equations_kinded (tenv : Typing_env.t) : Prop :=
  ∀ (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    type_equations_kinded tenv ty.

(* ======================================================================== *)
(* The kind of a [Simple]                                                   *)
(* ======================================================================== *)

(* A name is of a kind exactly when the type that the environment has
   for it is of that kind, or it is a symbol that is defined and has no
   type, and the kind is that of values. Whether the environment is
   bottom makes no difference. *)
Definition simple_of_kind_name_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (kind : flambda_kind),
    simple_of_kind tenv (simple_of_name n) kind
    <-> (∃ (ty : flambda_type)
           (binding_time_and_mode : Binding_time.With_name_mode.t),
           Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
           = Some (ty, binding_time_and_mode)
           ∧ FlambdaTypeGrammar.kind ty = kind)
        ∨ (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv) = None
           ∧ Typing_env.mem None tenv n = true
           ∧ kind = Value).

(* A [Simple] is of one kind at most. *)
Definition simple_of_kind_unique_statement : Prop :=
  ∀ (tenv : Typing_env.t) (s : simple) (kind kind' : flambda_kind),
    simple_of_kind tenv s kind -> simple_of_kind tenv s kind' -> kind = kind'.

(* A [Simple] keeps its kind from one environment to another in which
   every name that had a type has one of the same kind, every symbol
   that was defined is defined, and a symbol that had no type has none
   or one of the kind of values. *)
Definition simple_of_kind_kept_statement : Prop :=
  ∀ (tenv tenv' : Typing_env.t) (s : simple) (kind : flambda_kind),
    (∀ (n : name) (ty : flambda_type)
       (binding_time_and_mode : Binding_time.With_name_mode.t),
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
      = Some (ty, binding_time_and_mode) ->
      ∃ (ty' : flambda_type)
        (binding_time_and_mode' : Binding_time.With_name_mode.t),
        Keyed.lookup name_eqb n (Typing_env.names_to_types tenv')
        = Some (ty', binding_time_and_mode')
        ∧ FlambdaTypeGrammar.kind ty' = FlambdaTypeGrammar.kind ty) ->
    (∀ sym : symbol,
      among sym (Typing_env.defined_symbols tenv) = true ->
      among sym (Typing_env.defined_symbols tenv') = true) ->
    (∀ (sym : symbol) (ty : flambda_type)
       (binding_time_and_mode : Binding_time.With_name_mode.t),
      Keyed.lookup name_eqb (Name_symbol sym) (Typing_env.names_to_types tenv)
      = None ->
      Keyed.lookup name_eqb (Name_symbol sym) (Typing_env.names_to_types tenv')
      = Some (ty, binding_time_and_mode) ->
      FlambdaTypeGrammar.kind ty = Value) ->
    simple_of_kind tenv s kind ->
    simple_of_kind tenv' s kind.

(* ======================================================================== *)
(* What the functions do to the kind of a [Simple]                          *)
(* ======================================================================== *)

Definition make_bottom_simple_of_kind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (s : simple) (kind : flambda_kind),
    simple_of_kind (Typing_env.make_bottom tenv) s kind
    <-> simple_of_kind tenv s kind.

Definition increment_scope_simple_of_kind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (s : simple) (kind : flambda_kind),
    simple_of_kind (Typing_env.increment_scope tenv) s kind
    <-> simple_of_kind tenv s kind.

Definition closure_env_simple_of_kind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (s : simple) (kind : flambda_kind),
    simple_of_kind (Typing_env.closure_env tenv) s kind
    <-> simple_of_kind tenv s kind.

(* A symbol that is defined is of the kind of values, if it was of no
   kind. *)
Definition add_symbol_definition_simple_of_kind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (sym : symbol) (s : simple) (kind : flambda_kind),
    simple_of_kind (Typing_env.add_symbol_definition tenv sym) s kind
    <-> simple_of_kind tenv s kind
        ∨ (s = Symbol sym
           ∧ kind = Value
           ∧ Keyed.lookup name_eqb (Name_symbol sym)
               (Typing_env.names_to_types tenv)
             = None).

(* A variable that is defined is of the kind that it is defined with,
   whatever it was. *)
Definition add_variable_definition_simple_of_kind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (var : variable) (kind_of_var : flambda_kind)
    (name_mode : Name_mode.t) (s : simple) (kind : flambda_kind),
    simple_of_kind
      (Typing_env.add_variable_definition tenv var kind_of_var name_mode) s
      kind
    <-> (s = Var var ∧ kind = kind_of_var)
        ∨ (s <> Var var ∧ simple_of_kind tenv s kind).

(* A name that gets a type of the kind that it is of: nothing changes.
   The premise is the check that the callers in [Meet_env] make before
   they give a name a type ([Typing_env.find] with the kind of the
   type). *)
Definition replace_equation_simple_of_kind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (ty : flambda_type) (s : simple)
    (kind : flambda_kind),
    simple_of_kind tenv (simple_of_name n) (FlambdaTypeGrammar.kind ty) ->
    (simple_of_kind (Typing_env.replace_equation tenv n ty) s kind
     <-> simple_of_kind tenv s kind).

Definition add_alias_simple_of_kind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result) (s : simple)
    (kind : flambda_kind),
    Typing_env.add_alias tenv canonical_element1 canonical_element2
    = Some (Ok result) ->
    (simple_of_kind (Typing_env.add_alias_result_t result) s kind
     <-> simple_of_kind tenv s kind).

(* ======================================================================== *)
(* The functions keep the kinds of what is known to be equal                *)
(* ======================================================================== *)

Definition create_aliases_kinded_statement : Prop :=
  aliases_kinded Typing_env.create.

Definition make_bottom_aliases_kinded_statement : Prop :=
  ∀ tenv : Typing_env.t,
    aliases_kinded tenv -> aliases_kinded (Typing_env.make_bottom tenv).

Definition increment_scope_aliases_kinded_statement : Prop :=
  ∀ tenv : Typing_env.t,
    aliases_kinded tenv -> aliases_kinded (Typing_env.increment_scope tenv).

Definition closure_env_aliases_kinded_statement : Prop :=
  ∀ tenv : Typing_env.t,
    aliases_kinded tenv -> aliases_kinded (Typing_env.closure_env tenv).

(* A symbol is defined. In an environment that is well formed the
   aliases speak only of names that are bound, so a symbol that was of
   no kind is known to be equal to nothing but itself. *)
Definition add_symbol_definition_aliases_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (sym : symbol),
    typing_env_well_formed tenv = true ->
    aliases_kinded tenv ->
    aliases_kinded (Typing_env.add_symbol_definition tenv sym).

(* A variable is defined that is not bound, for the same reason. *)
Definition add_variable_definition_aliases_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (var : variable) (kind : flambda_kind)
    (name_mode : Name_mode.t),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var var) = false ->
    aliases_kinded tenv ->
    aliases_kinded
      (Typing_env.add_variable_definition tenv var kind name_mode).

(* A name gets a type of the kind that it is of. *)
Definition replace_equation_aliases_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (ty : flambda_type),
    simple_of_kind tenv (simple_of_name n) (FlambdaTypeGrammar.kind ty) ->
    aliases_kinded tenv ->
    aliases_kinded (Typing_env.replace_equation tenv n ty).

(* Two canonical elements of one kind are made aliases. *)
Definition add_alias_aliases_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result) (kind : flambda_kind),
    aliases_consistent (Typing_env.aliases tenv) ->
    is_canonical (Typing_env.aliases tenv) canonical_element1 ->
    is_canonical (Typing_env.aliases tenv) canonical_element2 ->
    simple_of_kind tenv canonical_element1 kind ->
    simple_of_kind tenv canonical_element2 kind ->
    Typing_env.add_alias tenv canonical_element1 canonical_element2
    = Some (Ok result) ->
    aliases_kinded tenv ->
    aliases_kinded (Typing_env.add_alias_result_t result).

(* ======================================================================== *)
(* Constants                                                                *)
(* ======================================================================== *)

(* Where two [Simple]s known to be equal are of one kind, a constant
   that is the canonical element of a [Simple] is of its kind. *)
Definition aliases_kinded_constants_of_kind_statement : Prop :=
  ∀ tenv : Typing_env.t,
    aliases_consistent (Typing_env.aliases tenv) ->
    aliases_kinded tenv ->
    constants_of_kind tenv.

(* ======================================================================== *)
(* What a kinded environment gives                                          *)
(* ======================================================================== *)

(* A type stays kinded where the [Simple]s keep their kinds. *)
Definition type_kinded_kept_statement : Prop :=
  ∀ (tenv tenv' : Typing_env.t) (ty : flambda_type),
    (∀ (s : simple) (kind : flambda_kind),
      simple_of_kind tenv s kind -> simple_of_kind tenv' s kind) ->
    type_kinded tenv ty ->
    type_kinded tenv' ty.

(* The type found for a name is kinded. *)
Definition find_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (kind : option flambda_kind)
    (ty : flambda_type),
    types_kinded tenv ->
    Typing_env.find tenv n kind = Some ty ->
    type_kinded tenv ty.

(* A symbol is of the kind of values and of no other. *)
Definition symbol_of_kind_value_statement : Prop :=
  ∀ (tenv : Typing_env.t) (s : symbol) (kind : flambda_kind),
    symbols_of_kind_value tenv ->
    simple_of_kind tenv (Symbol s) kind ->
    kind = Value.

(* The canonical element of a [Simple] of some kind is of that kind. *)
Definition canonical_ignoring_name_mode_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (s : simple) (kind : flambda_kind),
    typing_env_well_formed tenv = true ->
    aliases_kinded tenv ->
    simple_of_kind tenv s kind ->
    simple_of_kind tenv
      (Typing_env.get_canonical_simple_ignoring_name_mode tenv s) kind.

Definition typing_env_kinded_constants_of_kind_statement : Prop :=
  ∀ tenv : Typing_env.t,
    aliases_consistent (Typing_env.aliases tenv) ->
    typing_env_kinded tenv ->
    constants_of_kind tenv.

(* ======================================================================== *)
(* The functions keep an environment kinded                                 *)
(* ======================================================================== *)

Definition create_kinded_statement : Prop :=
  typing_env_kinded Typing_env.create.

Definition make_bottom_kinded_statement : Prop :=
  ∀ tenv : Typing_env.t,
    typing_env_kinded tenv -> typing_env_kinded (Typing_env.make_bottom tenv).

Definition increment_scope_kinded_statement : Prop :=
  ∀ tenv : Typing_env.t,
    typing_env_kinded tenv ->
    typing_env_kinded (Typing_env.increment_scope tenv).

Definition closure_env_kinded_statement : Prop :=
  ∀ tenv : Typing_env.t,
    typing_env_kinded tenv -> typing_env_kinded (Typing_env.closure_env tenv).

Definition add_symbol_definition_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (sym : symbol),
    typing_env_well_formed tenv = true ->
    typing_env_kinded tenv ->
    typing_env_kinded (Typing_env.add_symbol_definition tenv sym).

(* A variable is defined that is not bound. *)
Definition add_variable_definition_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (var : variable) (kind : flambda_kind)
    (name_mode : Name_mode.t),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var var) = false ->
    typing_env_kinded tenv ->
    typing_env_kinded
      (Typing_env.add_variable_definition tenv var kind name_mode).

(* A name gets a type that is kinded, of the kind that the name is
   of. *)
Definition replace_equation_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (ty : flambda_type),
    simple_of_kind tenv (simple_of_name n) (FlambdaTypeGrammar.kind ty) ->
    type_kinded tenv ty ->
    typing_env_kinded tenv ->
    typing_env_kinded (Typing_env.replace_equation tenv n ty).

(* Two canonical elements of one kind are made aliases. *)
Definition add_alias_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result) (kind : flambda_kind),
    aliases_consistent (Typing_env.aliases tenv) ->
    is_canonical (Typing_env.aliases tenv) canonical_element1 ->
    is_canonical (Typing_env.aliases tenv) canonical_element2 ->
    simple_of_kind tenv canonical_element1 kind ->
    simple_of_kind tenv canonical_element2 kind ->
    Typing_env.add_alias tenv canonical_element1 canonical_element2
    = Some (Ok result) ->
    typing_env_kinded tenv ->
    typing_env_kinded (Typing_env.add_alias_result_t result).

(* ======================================================================== *)
(* The equations inside types                                               *)
(* ======================================================================== *)

(* The equations inside a type stay kinded where the [Simple]s keep
   their kinds. *)
Definition type_equations_kinded_kept_statement : Prop :=
  ∀ (tenv tenv' : Typing_env.t) (ty : flambda_type),
    (∀ (s : simple) (kind : flambda_kind),
      simple_of_kind tenv s kind -> simple_of_kind tenv' s kind) ->
    type_equations_kinded tenv ty ->
    type_equations_kinded tenv' ty.

(* The equations inside the type found for a name are kinded. *)
Definition find_equations_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (kind : option flambda_kind)
    (ty : flambda_type),
    types_equations_kinded tenv ->
    Typing_env.find tenv n kind = Some ty ->
    type_equations_kinded tenv ty.

(* The functions keep the equations inside the types of an environment
   kinded. *)
Definition create_equations_kinded_statement : Prop :=
  types_equations_kinded Typing_env.create.

Definition make_bottom_equations_kinded_statement : Prop :=
  ∀ tenv : Typing_env.t,
    types_equations_kinded tenv ->
    types_equations_kinded (Typing_env.make_bottom tenv).

Definition increment_scope_equations_kinded_statement : Prop :=
  ∀ tenv : Typing_env.t,
    types_equations_kinded tenv ->
    types_equations_kinded (Typing_env.increment_scope tenv).

Definition closure_env_equations_kinded_statement : Prop :=
  ∀ tenv : Typing_env.t,
    types_equations_kinded tenv ->
    types_equations_kinded (Typing_env.closure_env tenv).

Definition add_symbol_definition_equations_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (sym : symbol),
    types_equations_kinded tenv ->
    types_equations_kinded (Typing_env.add_symbol_definition tenv sym).

(* A variable is defined that is not bound. *)
Definition add_variable_definition_equations_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (var : variable) (kind : flambda_kind)
    (name_mode : Name_mode.t),
    Typing_env.mem None tenv (Name_var var) = false ->
    types_equations_kinded tenv ->
    types_equations_kinded
      (Typing_env.add_variable_definition tenv var kind name_mode).

(* A name gets a type of the kind that the name is of, the equations
   inside which are kinded. *)
Definition replace_equation_equations_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (ty : flambda_type),
    simple_of_kind tenv (simple_of_name n) (FlambdaTypeGrammar.kind ty) ->
    type_equations_kinded tenv ty ->
    types_equations_kinded tenv ->
    types_equations_kinded (Typing_env.replace_equation tenv n ty).

Definition add_alias_equations_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result),
    Typing_env.add_alias tenv canonical_element1 canonical_element2
    = Some (Ok result) ->
    types_equations_kinded tenv ->
    types_equations_kinded (Typing_env.add_alias_result_t result).
