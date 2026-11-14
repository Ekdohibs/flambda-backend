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


(* Simplifying a primitive from the types of its arguments: the first
   rewrites.

   This file is to be reviewed: the definition and the statements,
   against middle_end/flambda2/simplify/simplify_primitive.ml,
   simplify_unary_primitive.ml and simplify_binary_primitive.ml.
   Refinement relative to a typing environment is
   FlambdaTypingEnvRefinementSpec.v; the provers and what their answers
   mean are FlambdaProvers.v and FlambdaProversSpec.v; the primitives
   are ../../rocq/FlambdaPrimitiveSpec.v.

   Simplify reads the type of each argument of a primitive from its
   typing environment, asks a prover about it, and follows the answer:
   the primitive becomes a constant, or the term becomes [Invalid].
   Each statement here says that one such step is right: under a
   typing environment, the [Let] of the primitive refines the [Let] of
   the constant, or is equivalent to [Invalid]
   ([expr_refines_under], [expr_equivalent_under]). So each is about
   every state of the machine whose variables and static environment
   satisfy the environment strictly. Nothing here is a pass, or runs
   in the validator.

   What is assumed.
   - Strict satisfaction ([env_satisfies_strictly]): some valuation
     gives every name that has a value in it the type that the
     environment has for the name, and agrees with the machine on the
     variables in scope at the normal name mode. (A name that occurs
     only in types, or is out of scope, has the value that the
     valuation chooses, whatever the machine has for it.) The
     statements of the provers are of that notion, and are false of
     the one that exempts the names that hold poison (decision D26;
     FlambdaProversSpec.v says what goes wrong). After the unboxing of
     variants the environments that Simplify builds are satisfied only
     with poison exempt, and these statements do not apply there.
   - For the provers of tagged immediates and for the prover of
     physical equality, that each type asked about and the types of
     the environment are well formed ([well_formed_input],
     FlambdaProversSpec.v, which says why). Simplify makes no such
     test.
   Nothing is assumed of the kinds of the term: where an argument is
   not of the kind that the primitive takes, the primitive has no
   value and the [Let] is undefined, which anything refines; and that
   is how Simplify treats it ([let_prim_kind_mismatch_statement]).

   The type of an argument ([argument_type]) is the one that
   [Simplify_simple.simplify_simple] answers: the alias type of the
   canonical element of the argument, at the normal name mode. The
   provers expand it. [simplify_simple] and [simplify_simples] compute
   it, with the canonical element, for the functions on terms that
   are built on these statements (FlambdaSimplifyPassSpec.v);
   [well_formed_input_b] is the test of what the provers of tagged
   immediates and the prover of physical equality ask of each type
   that they are given.

   What Simplify does, and what is here.
   - [Simplify_primitive.simplify_primitive] first replaces each
     argument by its canonical element
     ([let_prim_canonical_statement]). Where the type of an argument
     is not of the kind that the primitive takes it makes the term
     [Invalid] ([arg_kind_mismatch]):
     [let_prim_kind_mismatch_statement].
     Then it looks for the primitive among those already computed (not
     here), and then asks about the types.
   - The two conclusions: a defining expression that has no value but
     one constant, in the states that satisfy the environment, may be
     replaced by the constant ([let_constant_statement]); one that has
     no value there makes the [Let] equivalent to [Invalid]
     ([let_invalid_statement]). The statements below are these two,
     with what a prover's answer says.
   - [Is_int] without [variant_only] ([simplify_is_int]): where
     [Provers.prove_is_int] proves a boolean, the primitive is that
     boolean ([let_is_int_proved_statement]). Simplify leaves the
     primitive in the term and gives its variable the type of the
     constant, so that the uses of the variable become the constant
     and the [Let], then unused, is removed. The statement is that the
     primitive may be replaced by the constant; with
     [let_simple_refines_under_statement]
     (FlambdaTypingEnvRefinementSpec.v) the body is then rewritten
     with the variable equal to the constant, as in the examples.
   - [Get_tag] ([simplify_get_tag]): where [Provers.prove_is_int]
     proves that the argument is a tagged immediate, the term is
     [Invalid] ([let_get_tag_of_immediate_statement]).
   - [Is_int] with [variant_only], [Is_null], and [Get_tag] otherwise
     ([simplify_relational_primitive]): Simplify does not ask a prover
     here. It gives the variable bound the type that says that it is
     the result of the relation on the argument, and the argument a
     type that names the variable. Three statements say that the
     value of the primitive has that type
     ([is_int_relation_type_statement],
     [is_null_relation_type_statement],
     [get_tag_relation_type_statement]); the one for [Get_tag] needs
     the tag of the block to be a tag, which the machine does not
     test and a run in range has (FlambdaTypingEnvRangeSpec.v). That
     the environment with the two equations is satisfied is for the
     statements about the meet, which adds them.
   - Then Simplify reifies the type of the variable
     ([Reification.try_to_reify], [Reify.reify]): for a type of naked
     immediates it asks [Provers.meet_naked_immediates], which reads
     the relation by asking of the type of the argument what
     [Provers.meet_is_int_variant_only], [Provers.prove_get_tag] and
     [Provers.meet_is_null] ask; if the answer is one number the
     primitive becomes that constant, and if there is none the term
     becomes [Invalid]. The statements here are of those three provers
     asked directly about the type of the argument, in the environment
     before the [Let] ([let_is_int_variant_only_statement],
     [let_get_tag_proved_statement], [let_is_null_statement]). They
     say that the conclusion is right where those provers answer; they
     do not say that this is the test that Simplify makes, which goes
     through the meet and through reification (the workstreams on meet
     and on the provers). The one for [Get_tag] needs no premise on
     tags: the prover's answer and the primitive are about the same
     block.
   - A comparison of two tagged immediates
     ([Simplify_binary_primitive], the functor [Binary_arith_like] at
     [Int_ops_for_binary_comp]): each argument is asked
     [Provers.meet_equals_tagged_immediates]. Where both answer
     numbers, and the comparison is the same number for every pair of
     them, the primitive is that number; where either answers that
     there is no tagged immediate, the term is [Invalid]
     ([let_int_comp_tagged_statement]). A comparison with a constant
     is the case where one argument is the constant, whose type has
     the one number. (Simplify gives up beyond ten numbers on a side;
     the statement has no such limit. Before it asks, it looks for a
     comparison of the result of a comparison with zero
     ([recover_comparison_primitive]): not here.)
   - [Phys_equal] ([simplify_phys_equal]): the two types are given to
     [Provers.prove_physical_equality]; where it proves that the two
     arguments are the same value, or that they are not, the primitive
     is that boolean, or the other one for [Phys_neq]
     ([let_phys_equal_statement]). A refinement: the primitive of the
     model has no value on two allocated values. (Where the prover
     does not know, Simplify records that the result is 0 or 1: not
     here.)
   - Two small terms are rewritten with these, from the environment
     that knows nothing ([example_is_int_statement],
     [example_int_comp_statement]).

   Not here: the other primitives; what Simplify records of the result
   where it does not replace the primitive (the type of the result
   variable); the equations that a relational primitive adds.

   The proofs are in FlambdaSimplifyPrimitivesProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaProvers FlambdaProversSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The type of an argument                                                  *)
(* ======================================================================== *)

(* The type that Simplify has for an argument of a primitive
   ([Simplify_simple.simplify_simple]): the argument may occur in
   terms, and the type is the one that the environment answers for it
   at the normal name mode, the alias type of its canonical element. *)
Definition argument_type (tenv : Typing_env.t) (arg : simple)
    (ty : flambda_type) : Prop :=
  Typing_env.mem_simple (Some Name_mode.normal) tenv arg = true
  ∧ ∃ canonical : simple,
      Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) arg
      = Some (Some (ty, canonical)).

(* In a state that satisfies the environment, the value of an argument
   has the type of the argument, under a valuation that satisfies the
   environment and agrees with the machine. *)
Definition argument_type_satisfied_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (arg : simple)
    (ty : flambda_type) (v : value),
    env_satisfies_strictly tenv statics vars ->
    argument_type tenv arg ty ->
    eval_simple vars arg = Some v ->
    ∃ valuation : env,
      typing_env_satisfied_strictly statics valuation tenv
      ∧ agrees_in_terms tenv valuation vars
      ∧ type_satisfied statics valuation v ty.

(* ======================================================================== *)
(* The type of an argument, computed                                        *)
(* ======================================================================== *)

(* [Simplify_simple.simplify_simple] at the normal name mode: the type
   of a [Simple] that occurs in the term, and its canonical element.
   [None] where the environment does not know the [Simple] at that
   name mode, which is a fatal error of the compiler. Where it answers
   a type, that is the type of the argument ([argument_type]). *)
Definition simplify_simple (tenv : Typing_env.t) (s : simple)
    : option (flambda_type * simple) :=
  if Typing_env.mem_simple (Some Name_mode.normal) tenv s
  then
    match
      Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s
    with
    | Some (Some answer) => Some answer
    | Some None | None => None
    end
  else None.

Fixpoint simplify_simples (tenv : Typing_env.t) (args : list simple)
    : option (list (flambda_type * simple)) :=
  match args with
  | [] => Some []
  | arg :: args =>
    let* answer := simplify_simple tenv arg in
    let* answers := simplify_simples tenv args in
    Some (answer :: answers)
  end.

(* The type asked about and the types of the environment are well
   formed: [well_formed_input] of FlambdaProversSpec.v, as a test. *)
Definition well_formed_input_b (tenv : Typing_env.t) (ty : flambda_type)
    : bool :=
  types_well_formed_b tenv && type_well_formed ty.

(* ======================================================================== *)
(* The arguments, and the two conclusions                                   *)
(* ======================================================================== *)

(* The arguments of a primitive may be replaced by their canonical
   elements. *)
Definition let_prim_canonical_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind) (p : prim)
    (args args' : list simple) (body : expr),
    Forall2 (canonical_in_terms tenv) args args' ->
    expr_equivalent_under tenv (Let x declared (Prim p args) body)
      (Let x declared (Prim p args') body).

(* [Simplify_primitive.arg_kind_mismatch]: where the types of the
   arguments are not of the kinds that the primitive takes, the term is
   [Invalid]. (So it is where the arguments are not as many as the
   primitive takes, which the compiler's terms cannot be.) This is not
   said of a primitive of the heap: [prim_args_well_kinded] knows the
   kinds of the arguments of none of them, and answers no whatever
   they are given, while the machine runs them. *)
Definition let_prim_kind_mismatch_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind) (p : prim)
    (args : list simple) (tys : list flambda_type) (body : expr),
    (∀ q : heap_primitive, p <> Heap q) ->
    Forall2 (argument_type tenv) args tys ->
    prim_args_well_kinded p (map FlambdaTypeGrammar.kind tys) = false ->
    expr_equivalent_under tenv (Let x declared (Prim p args) body) Invalid.

(* A defining expression that has no value but one constant, in the
   states that satisfy the environment, and that leaves the heap and
   the read record as they came, may be replaced by the constant. The
   machine runs the defining expression on its heap
   (FlambdaSemanticsSpec.exec_named): its value is the value of that
   run, on whatever heap and with whatever record. The heap is asked
   to be as it came because a primitive of the heap may answer a
   constant and write (a store answers the unit value), and the record
   because the [Let] of the constant records nothing. This is a
   refinement: the defining expression may have no value. *)
Definition let_constant_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (defining_expr : named) (c : const) (body : expr),
    (∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
       (vars : env) (v : value),
      env_satisfies_strictly tenv statics vars ->
      exec_named statics hp rd vars defining_expr = Some (v, hp', rd') ->
      v = Value_const c ∧ hp' = hp ∧ rd' = rd) ->
    expr_refines_under tenv (Let x declared defining_expr body)
      (Let x declared (Simple (Const c)) body).

(* A [Let] whose defining expression has no value, in the states that
   satisfy the environment, on whatever heap and with whatever record
   the machine runs it, is [Invalid] there. *)
Definition let_invalid_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (defining_expr : named) (body : expr),
    (∀ (statics : static_env) (hp : heap) (rd : read_record) (vars : env),
      env_satisfies_strictly tenv statics vars ->
      exec_named statics hp rd vars defining_expr = None) ->
    expr_equivalent_under tenv (Let x declared defining_expr body) Invalid.

(* ======================================================================== *)
(* [Is_int] and [Get_tag], through [Provers.prove_is_int]                   *)
(* ======================================================================== *)

(* [Simplify_unary_primitive.simplify_is_int], without [variant_only]:
   where the prover proves a boolean, the primitive is that boolean. *)
Definition let_is_int_proved_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (arg : simple) (ty : flambda_type) (b : bool) (body : expr),
    argument_type tenv arg ty ->
    Provers.prove_is_int tenv ty = Some (Proof_of_property.Proved b) ->
    expr_refines_under tenv
      (Let x declared (Prim (Unary (Is_int false)) [arg]) body)
      (Let x declared (Simple (Const (Naked_immediate (if b then 1 else 0))))
         body).

(* [Simplify_unary_primitive.simplify_get_tag]: where the prover proves
   that the argument is a tagged immediate, the term is [Invalid]. *)
Definition let_get_tag_of_immediate_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (arg : simple) (ty : flambda_type) (body : expr),
    argument_type tenv arg ty ->
    Provers.prove_is_int tenv ty = Some (Proof_of_property.Proved true) ->
    expr_equivalent_under tenv
      (Let x declared (Prim (Unary Get_tag) [arg]) body) Invalid.

(* ======================================================================== *)
(* The relational primitives                                                *)
(* ======================================================================== *)

(* The value of [Is_int] with [variant_only] on the value of a name has
   the type that says that it is [Is_int] of the name
   ([Simplify_unary_primitive.simplify_relational_primitive] with
   [add_is_int_relation]). Without [variant_only] it has not: the
   primitive answers 0 on a closure, which the relation relates to
   nothing. *)
Definition is_int_relation_type_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (scrutinee : name)
    (w r : value),
    value_of_name valuation scrutinee = Some w ->
    eval_prim statics (Unary (Is_int true)) [w] = Some r ->
    type_satisfied statics valuation r (is_int_for_scrutinee scrutinee).

(* The same of [Is_null] ([add_is_null_relation]). *)
Definition is_null_relation_type_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (scrutinee : name)
    (w r : value),
    value_of_name valuation scrutinee = Some w ->
    eval_prim statics (Unary Is_null) [w] = Some r ->
    type_satisfied statics valuation r (is_null_for_scrutinee scrutinee).

(* And of [Get_tag] ([add_get_tag_relation]), where the tag of the
   block is a tag. The machine does not test that, and the relation
   relates a block to its tag only if it is one. *)
Definition get_tag_relation_type_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (block : name) (w r : value),
    value_of_name valuation block = Some w ->
    eval_prim statics (Unary Get_tag) [w] = Some r ->
    (∀ (tag : Z) (fields : list value),
      block_of statics w = Some (tag, fields) -> is_tag tag) ->
    type_satisfied statics valuation r (get_tag_for_block block).

(* What reading the relation comes to, asked of the type of the
   argument. *)

(* [Is_int] with [variant_only]: where
   [Provers.meet_is_int_variant_only] answers a boolean the primitive
   is that boolean, and where it answers that no value of the type is a
   tagged immediate or a block the term is [Invalid]. *)
Definition let_is_int_variant_only_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (arg : simple) (ty : flambda_type) (body : expr),
    argument_type tenv arg ty ->
    match Provers.meet_is_int_variant_only tenv ty with
    | Some (Meet_shortcut.Known_result b) =>
      expr_refines_under tenv
        (Let x declared (Prim (Unary (Is_int true)) [arg]) body)
        (Let x declared
           (Simple (Const (Naked_immediate (if b then 1 else 0)))) body)
    | Some Meet_shortcut.Invalid =>
      expr_equivalent_under tenv
        (Let x declared (Prim (Unary (Is_int true)) [arg]) body) Invalid
    | Some Meet_shortcut.Need_meet | None => True
    end.

(* [Get_tag]: where [Provers.prove_get_tag] proves that every value of
   the type is a block of one tag, the primitive is that tag. *)
Definition let_get_tag_proved_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (arg : simple) (ty : flambda_type) (tags : list Z) (tag : Z)
    (body : expr),
    argument_type tenv arg ty ->
    Provers.prove_get_tag tenv ty = Some (Proof_of_property.Proved tags) ->
    (∀ tag' : Z, tag' ∈ tags -> tag' = tag) ->
    expr_refines_under tenv
      (Let x declared (Prim (Unary Get_tag) [arg]) body)
      (Let x declared (Simple (Const (Naked_immediate tag))) body).

(* [Is_null]: where [Provers.meet_is_null] answers a boolean the
   primitive is that boolean, and where it answers that no value of the
   type is of kind value the term is [Invalid]. *)
Definition let_is_null_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (arg : simple) (ty : flambda_type) (body : expr),
    argument_type tenv arg ty ->
    match Provers.meet_is_null tenv ty with
    | Some (Meet_shortcut.Known_result b) =>
      expr_refines_under tenv
        (Let x declared (Prim (Unary Is_null) [arg]) body)
        (Let x declared
           (Simple (Const (Naked_immediate (if b then 1 else 0)))) body)
    | Some Meet_shortcut.Invalid =>
      expr_equivalent_under tenv
        (Let x declared (Prim (Unary Is_null) [arg]) body) Invalid
    | Some Meet_shortcut.Need_meet | None => True
    end.

(* ======================================================================== *)
(* A comparison of tagged immediates                                        *)
(* ======================================================================== *)

(* [Simplify_binary_primitive], for [Int_comp] on tagged immediates:
   where [Provers.meet_equals_tagged_immediates] answers numbers for
   both arguments, and the comparison is the same number for every
   pair of them, the primitive is that number; where it answers for
   either that no value of the type is a tagged immediate, the term is
   [Invalid]. *)
Definition let_int_comp_tagged_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (behaviour : int_comparison_behaviour) (arg1 arg2 : simple)
    (ty1 ty2 : flambda_type) (body : expr),
    argument_type tenv arg1 ty1 ->
    argument_type tenv arg2 ty2 ->
    well_formed_input tenv ty1 ->
    well_formed_input tenv ty2 ->
    let comparison :=
      Prim (Binary (Int_comp Standard_tagged_immediate behaviour))
        [arg1; arg2]
    in
    match
      Provers.meet_equals_tagged_immediates tenv ty1,
      Provers.meet_equals_tagged_immediates tenv ty2
    with
    | Some (Meet_shortcut.Known_result imms1),
      Some (Meet_shortcut.Known_result imms2) =>
      ∀ result : Z,
        (∀ i1 i2 : Z,
          i1 ∈ imms1 ->
          i2 ∈ imms2 ->
          eval_int_comp (int_width Standard_tagged_immediate) behaviour i1 i2
          = result) ->
        expr_refines_under tenv (Let x declared comparison body)
          (Let x declared (Simple (Const (Naked_immediate result))) body)
    | Some Meet_shortcut.Invalid, _ | _, Some Meet_shortcut.Invalid =>
      expr_equivalent_under tenv (Let x declared comparison body) Invalid
    | _, _ => True
    end.

(* ======================================================================== *)
(* Physical equality                                                        *)
(* ======================================================================== *)

(* [Simplify_binary_primitive.simplify_phys_equal]: where
   [Provers.prove_physical_equality] proves a boolean of the types of
   the two arguments, the primitive is that boolean for [Phys_eq] and
   the other one for [Phys_neq]. The types are well formed, as the
   statement of the prover asks. *)
Definition let_phys_equal_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable) (declared : flambda_kind)
    (op : equality_comparison) (arg1 arg2 : simple) (ty1 ty2 : flambda_type)
    (b : bool) (body : expr),
    argument_type tenv arg1 ty1 ->
    argument_type tenv arg2 ty2 ->
    well_formed_input tenv ty1 ->
    well_formed_input tenv ty2 ->
    Provers.prove_physical_equality tenv ty1 ty2
    = Some (Proof_of_property.Proved b) ->
    let result := match op with Phys_eq => b | Phys_neq => negb b end in
    expr_refines_under tenv
      (Let x declared (Prim (Binary (Phys_equal op)) [arg1; arg2]) body)
      (Let x declared
         (Simple (Const (Naked_immediate (if result then 1 else 0)))) body).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The variables are those of FlambdaTypingEnvExamplesSpec.v. Each term
   is closed but for its continuations, so Simplify starts on it with
   the environment that knows nothing. *)

(* [let x = 5 in let i = Is_int x in switch i with 0 -> k0 | 1 -> k1].
   [x] is made equal to the constant; the type of the argument of
   [Is_int] is the alias type of the constant, of which the prover
   proves that it is a tagged immediate; [i] is made equal to 1, and
   the [Switch] is its arm. *)
Definition example_is_int_statement : Prop :=
  ∀ k0 k1 : continuation,
    expr_refines
      (Let example_x Value (Simple (Const (Tagged_immediate 5)))
         (Let example_i (Naked_number Kind_naked_immediate)
            (Prim (Unary (Is_int false)) [Var example_x])
            (Switch (Var example_i)
               [(0, Apply_cont_expr k0 [] None);
                (1, Apply_cont_expr k1 [] None)])))
      (Let example_x Value (Simple (Const (Tagged_immediate 5)))
         (Let example_i (Naked_number Kind_naked_immediate)
            (Simple (Const (Naked_immediate 1)))
            (Apply_cont (Apply_cont_expr k1 [] None)))).

(* [let x = 5 in let i = (x < 7) in switch i with 0 -> k0 | 1 -> k1],
   the comparison being of tagged immediates, signed. *)
Definition example_int_comp_statement : Prop :=
  ∀ k0 k1 : continuation,
    expr_refines
      (Let example_x Value (Simple (Const (Tagged_immediate 5)))
         (Let example_i (Naked_number Kind_naked_immediate)
            (Prim
               (Binary
                  (Int_comp Standard_tagged_immediate
                     (Yielding_bool (Int_lt Signed))))
               [Var example_x; Const (Tagged_immediate 7)])
            (Switch (Var example_i)
               [(0, Apply_cont_expr k0 [] None);
                (1, Apply_cont_expr k1 [] None)])))
      (Let example_x Value (Simple (Const (Tagged_immediate 5)))
         (Let example_i (Naked_number Kind_naked_immediate)
            (Simple (Const (Naked_immediate 1)))
            (Apply_cont (Apply_cont_expr k1 [] None)))).
