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


(* Simplifying a load or a projection from the type of its argument:
   the rewrites to a [Simple], and the type of a block that is made.

   This file is to be reviewed: the definitions and the statements,
   against middle_end/flambda2/simplify/simplify_unary_primitive.ml
   ([simplify_immutable_block_load], [simplify_project_function_slot],
   [simplify_project_value_slot]) and simplify_variadic_primitive.ml
   ([simplify_make_block]). It follows
   FlambdaSimplifyPrimitivesSpec.v, which has the type of an argument
   ([argument_type]), the rewrites to a constant and to [Invalid], and
   what is assumed: strict satisfaction of the environment.

   Where a prover answers a [Simple] for a field of a block, for a
   closure of a set or for the contents of a value slot, Simplify
   binds the variable of the [Let] to that [Simple] instead of the
   load or the projection, and gives the variable the alias type of
   it. Each statement here says that the first of these is right:
   under the typing environment, the [Let] of the load refines the
   [Let] of the [Simple]. (The second is then
   [let_simple_refines_under_statement],
   FlambdaTypingEnvRefinementSpec.v, as in the example.) Where the
   prover answers that the type has no value to load from, the [Let]
   is equivalent to [Invalid].

   The [Simple] may occur in the term. The statements of the provers
   say that the [Simple] answered has the right value in a valuation
   that satisfies the environment; a valuation gives values to names
   that no term can mention, and agrees with the machine only on the
   variables in scope at the normal name mode
   ([agrees_in_terms], FlambdaTypingEnvSpec.v). Simplify asks the
   prover at the name mode of the variable of the [Let], which in the
   model is the normal one. What the prover answers there is the
   canonical element that the environment answers, at that name mode,
   for the alias that the type of the field, of the closure or of the
   slot is; and such an element is a constant, a symbol or a variable
   in scope at that name mode ([canonical_simple_in_terms_statement],
   [provers_answer_in_terms_statement]). So it has in the machine the
   value that the valuation gives it
   ([simple_in_terms_agrees_statement]).

   What Simplify does, and what is here.
   - [Block_load] that is not marked [Mutable]
     ([simplify_immutable_block_load0]):
     [Provers.meet_block_field_simple] is asked about the type of the
     block, at the kind that the load gives its result, which for a
     block of values is the kind value, and at the field
     ([let_block_load_statement]). A load marked [Mutable] is
     [simplify_mutable_block_load], which asks no prover, and is not
     here: the statement has the premise that the load is not marked
     so. (Every block of the model is immutable, so the rewrite would
     be right of the model for a load marked [Mutable] too; it is not
     what Simplify does, and would not be right of blocks that can be
     written.)
   - [Project_function_slot] ([simplify_project_function_slot]):
     [Provers.meet_project_function_slot_simple], asked about the slot
     projected to ([let_project_function_slot_statement]).
   - [Project_value_slot] ([simplify_project_value_slot]):
     [Provers.meet_project_value_slot_simple], asked at the kind of
     the value slot. A value slot of the model has no kind, and the
     prover's statement is of the closures whose slot has contents of
     the kind asked (FlambdaProversSpec.v, discrepancy T6-4). So the
     statement here has the premise that the projection, where it has
     a value, has one of that kind
     ([let_project_value_slot_statement]). (In the compiler the
     [Simple] answered may carry a coercion, which Simplify then
     simplifies; the model has no coercions.)
   - Where the prover answers neither, Simplify meets the type of the
     argument with a type that names the variable of the [Let]
     ([Simplify_common.simplify_projection]): not here.
   - [Make_block] of an immutable block of values
     ([simplify_make_block]): the primitive stays, and its variable is
     given the type of the blocks of that tag whose fields have the
     types of the arguments, which are aliases. The value of the
     primitive has that type ([make_block_typed_statement]), so the
     body of the [Let] is rewritten under the environment with that
     equation ([let_make_block_statement]). The tag is to be one of a
     block that is scanned, as the compiler's are; the machine makes a
     block of any tag. (Simplify also meets each argument that is a
     variable with the subkind that the primitive gives the field;
     the model has no subkinds.)
   - An example with both: the first field of a pair that was just
     made is the first argument ([example_first_of_pair_statement]).

   Not here: [Unbox_number] and [Untag_immediate], which Simplify
   treats through the meet; the symbol projections and the equations
   of common subexpressions that Simplify records at a load.

   The proofs are in FlambdaSimplifyProjectionsProofs.v, which needs
   no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvExamplesSpec
  FlambdaTypingEnvRefinementSpec FlambdaProvers
  FlambdaProversSpec FlambdaSimplifyPrimitivesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* A [Simple] that may occur in terms                                       *)
(* ======================================================================== *)

(* A [Simple] may occur in a term simplified under the environment: it
   is a constant, a symbol, or a variable in scope at the normal name
   mode. *)
Definition simple_in_terms (tenv : Typing_env.t) (s : simple) : Prop :=
  ∀ x : variable, s = Var x -> in_scope_in_terms tenv x.

(* Such a [Simple] has in the machine the value that a valuation that
   agrees with the machine gives it. *)
Definition simple_in_terms_agrees_statement : Prop :=
  ∀ (tenv : Typing_env.t) (valuation vars : env) (s : simple),
    agrees_in_terms tenv valuation vars ->
    simple_in_terms tenv s ->
    eval_simple vars s = eval_simple valuation s.

(* The canonical element that the environment answers at the normal
   name mode, for a [Simple] whose name mode it reads itself, may
   occur in terms. *)
Definition canonical_simple_in_terms_statement : Prop :=
  ∀ (tenv : Typing_env.t) (s canonical : simple),
    Typing_env.get_canonical_simple_exn tenv (Some Name_mode.normal) None s
    = Some (Some canonical) ->
    simple_in_terms tenv canonical.

(* The [Simple] that a meet shortcut answers may occur in terms. *)
Definition answer_in_terms
    (prover : Typing_env.t -> flambda_type -> option (Meet_shortcut.t simple))
    : Prop :=
  ∀ (tenv : Typing_env.t) (t : flambda_type) (s : simple),
    prover tenv t = Some (Meet_shortcut.Known_result s) ->
    simple_in_terms tenv s.

(* The three provers, asked at the normal name mode. *)
Definition provers_answer_in_terms_statement : Prop :=
  (∀ (field_kind : flambda_kind) (field_index : nat),
    answer_in_terms
      (fun tenv t =>
         Provers.meet_block_field_simple tenv Name_mode.normal field_kind t
           field_index))
  ∧ (∀ move_to : function_slot,
      answer_in_terms
        (fun tenv t =>
           Provers.meet_project_function_slot_simple tenv Name_mode.normal t
             move_to))
  ∧ ∀ (slot : value_slot) (value_slot_kind : flambda_kind),
      answer_in_terms
        (fun tenv t =>
           Provers.meet_project_value_slot_simple tenv Name_mode.normal t
             slot value_slot_kind).

(* ======================================================================== *)
(* The conclusion                                                           *)
(* ======================================================================== *)

(* A defining expression that has no value but that of a [Simple], in
   the states that satisfy the environment, and that leaves the heap
   and the read record as they came, may be replaced by the [Simple].
   Its value is that of the run of the machine on its heap
   (FlambdaSemanticsSpec.exec_named), on whatever heap and with
   whatever record. This is a refinement: the defining expression may
   have no value. ([let_constant_statement] of
   FlambdaSimplifyPrimitivesSpec.v is the case of a constant, and says
   why the heap and the record are asked to be as they came.) *)
Definition let_simple_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (defining_expr : named) (s : simple) (body : expr),
    (∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
       (vars : env) (v : value),
      env_satisfies_strictly tenv statics vars ->
      exec_named statics hp rd vars defining_expr = Some (v, hp', rd') ->
      eval_simple vars s = Some v ∧ hp' = hp ∧ rd' = rd) ->
    expr_refines_under tenv (Let x declared defining_expr body)
      (Let x declared (Simple s) body).

(* ======================================================================== *)
(* A load from a block                                                      *)
(* ======================================================================== *)

(* [Simplify_unary_primitive.simplify_immutable_block_load0]: where
   [Provers.meet_block_field_simple] answers a [Simple] the load is
   that [Simple], and where it answers that no value of the type is a
   block with that field, of kind value, the term is [Invalid]. (A
   load at a field below zero has no value; the prover is then asked
   about the field zero, and either conclusion holds.) The load is
   not marked [Mutable]: Simplify sends such a load elsewhere
   ([simplify_mutable_block_load]). *)
Definition let_block_load_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (kind : block_access_kind) (mut : mutability) (field : Z) (arg : simple)
    (ty : flambda_type) (body : expr),
    mut <> Mutable ->
    argument_type tenv arg ty ->
    let load := Prim (Unary (Block_load kind mut field)) [arg] in
    match
      Provers.meet_block_field_simple tenv Name_mode.normal Value ty
        (Z.to_nat field)
    with
    | Some (Meet_shortcut.Known_result s) =>
      expr_refines_under tenv (Let x declared load body)
        (Let x declared (Simple s) body)
    | Some Meet_shortcut.Invalid =>
      expr_equivalent_under tenv (Let x declared load body) Invalid
    | Some Meet_shortcut.Need_meet | None => True
    end.

(* ======================================================================== *)
(* The projections from a closure                                           *)
(* ======================================================================== *)

(* [Simplify_unary_primitive.simplify_project_function_slot]: where
   [Provers.meet_project_function_slot_simple] answers a [Simple] the
   projection is that [Simple], and where it answers that no value of
   the type is a closure of a set that has the slot, the term is
   [Invalid]. *)
Definition let_project_function_slot_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (move_from move_to : function_slot) (closure : simple)
    (ty : flambda_type) (body : expr),
    argument_type tenv closure ty ->
    let projection := Project_function_slot move_from move_to closure in
    match
      Provers.meet_project_function_slot_simple tenv Name_mode.normal ty
        move_to
    with
    | Some (Meet_shortcut.Known_result s) =>
      expr_refines_under tenv (Let x declared projection body)
        (Let x declared (Simple s) body)
    | Some Meet_shortcut.Invalid =>
      expr_equivalent_under tenv (Let x declared projection body) Invalid
    | Some Meet_shortcut.Need_meet | None => True
    end.

(* [Simplify_unary_primitive.simplify_project_value_slot], of a
   projection that has no value but one of the kind of the value slot:
   where [Provers.meet_project_value_slot_simple] answers a [Simple]
   the projection is that [Simple], and where it answers that no value
   of the type is a closure whose slot has contents of that kind, the
   term is [Invalid]. *)
(* CR mshinwell: the premise on the kind is not one that Simplify
   checks, and nothing in the model gives it: a value slot of the
   model has no kind, and a set of closures may have contents of any
   kind in a slot. Without it the conclusion on [Invalid] is false:
   the prover, asked at the kind value about a closure whose slot
   holds a naked float, answers [Invalid], and the projection has
   that float for value (FlambdaProversExamplesSpec.v). The premise
   holds where the terms respect the kinds of the value slots, which
   is for the rulings on kinds. Discrepancy T6-4. *)
Definition let_project_value_slot_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (project_from : function_slot) (slot : value_slot)
    (value_slot_kind : flambda_kind) (closure : simple) (ty : flambda_type)
    (body : expr),
    argument_type tenv closure ty ->
    let projection := Project_value_slot_from project_from slot closure in
    (∀ (statics : static_env) (vars : env) (v : value),
      env_satisfies_strictly tenv statics vars ->
      eval_named statics vars projection = Some v ->
      kind_of_value v = value_slot_kind) ->
    match
      Provers.meet_project_value_slot_simple tenv Name_mode.normal ty slot
        value_slot_kind
    with
    | Some (Meet_shortcut.Known_result s) =>
      expr_refines_under tenv (Let x declared projection body)
        (Let x declared (Simple s) body)
    | Some Meet_shortcut.Invalid =>
      expr_equivalent_under tenv (Let x declared projection body) Invalid
    | Some Meet_shortcut.Need_meet | None => True
    end.

(* ======================================================================== *)
(* A block that is made                                                     *)
(* ======================================================================== *)

(* The type of an argument in the machine: the value that the machine
   has for the argument has the type of the argument under the
   variables of the machine, and the type speaks only of variables in
   scope. (The type is the alias type of a [Simple] that may occur in
   terms; [argument_type_satisfied_statement] is the same under a
   valuation.) Only the aliases are read, so this holds where poison
   is exempt too. *)
Definition argument_type_in_machine_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (arg : simple)
    (ty : flambda_type) (v : value),
    env_satisfies tenv statics vars ->
    argument_type tenv arg ty ->
    eval_simple vars arg = Some v ->
    type_satisfied statics vars v ty.

Definition argument_type_in_scope_statement : Prop :=
  ∀ (tenv : Typing_env.t) (arg : simple) (ty : flambda_type) (y : variable),
    argument_type tenv arg ty ->
    y ∈ free_variables ty ->
    in_scope_in_terms tenv y.

(* The type that [Simplify_variadic_primitive.simplify_make_block]
   gives the variable of a [Make_block] of an immutable block of
   values, allocated on the heap: the blocks of the tag whose fields
   have the types of the arguments. *)
Definition make_block_type (tag : Z) (tys : list flambda_type)
    : flambda_type :=
  immutable_block false tag (Scannable Value_only) Alloc_mode.Heap tys.

(* The value of the primitive has that type, where the tag is one of a
   block that is scanned. *)
Definition make_block_typed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (tag : Z)
    (args : list simple) (tys : list flambda_type) (v : value),
    env_satisfies tenv statics vars ->
    Forall2 (argument_type tenv) args tys ->
    tag < no_scan_tag ->
    eval_named statics vars
      (Prim (Variadic (Make_block (Block_kind_values tag))) args)
    = Some v ->
    type_satisfied statics vars v (make_block_type tag tys).

(* So the body of the [Let] is rewritten under the environment in
   which the variable is defined with that type. *)
Definition let_make_block_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind) (tag : Z)
    (args : list simple) (tys : list flambda_type) (body body' : expr),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    Forall2 (argument_type tenv) args tys ->
    tag < no_scan_tag ->
    let make_block :=
      Prim (Variadic (Make_block (Block_kind_values tag))) args
    in
    expr_refines_under
      (Typing_env.replace_equation
         (Typing_env.add_variable_definition tenv x Value Name_mode.normal)
         (Name_var x) (make_block_type tag tys))
      body body' ->
    expr_refines_under tenv (Let x declared make_block body)
      (Let x declared make_block body').

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The variables are those of FlambdaTypingEnvExamplesSpec.v. *)

(* The environment in which [x] and [y] are defined at the kind value
   and nothing more is known of them, as for the parameters of a
   continuation or of a function. *)
Definition example_pair_env : Typing_env.t :=
  Typing_env.add_definitions_of_params Typing_env.create
    [(example_x, Value); (example_y, Value)].

(* [let p = Make_block 0 [x; y] in let q = Block_load p 0 in
   apply_cont k q], which is the first component of the pair of [x]
   and [y]. [p] is given the type of the blocks of tag 0 whose two
   fields are [x] and [y]; of the type of [p] the prover answers that
   the field 0 is [x]; [q] is made equal to [x], and replaced by it in
   the body. (The two [Let]s are then unused.) *)
Definition example_first_of_pair_statement : Prop :=
  ∀ k : continuation,
    let pair :=
      Prim (Variadic (Make_block (Block_kind_values 0)))
        [Var example_x; Var example_y]
    in
    expr_refines_under example_pair_env
      (Let example_p Value pair
         (Let example_q Value
            (Prim
               (Unary
                  (Block_load (Block_access_values (Some 0) (Some 2) Any_value)
                     Immutable 0))
               [Var example_p])
            (Apply_cont (Apply_cont_expr k [Var example_q] None))))
      (Let example_p Value pair
         (Let example_q Value (Simple (Var example_x))
            (Apply_cont (Apply_cont_expr k [Var example_x] None)))).
