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


(* Proofs of the statements of FlambdaSimplifyPrimitivesSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaHeapSpec
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaTypingEnvRefinementProofs FlambdaProvers
  FlambdaProversSpec FlambdaProversProofs FlambdaSimplifyPrimitivesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The type of an argument                                                  *)
(* ======================================================================== *)

Theorem argument_type_satisfied : argument_type_satisfied_statement.
Proof.
  intros tenv statics vars arg ty v [Hbound [valuation [Hsatisfied Hagrees]]]
    [Hmem [canonical Htype]] Heval.
  exists valuation. split; [ exact Hsatisfied | ].
  split; [ exact Hagrees | ].
  apply (proj2 (type_simple_in_term_satisfied (fun _ => False) statics
                  valuation tenv (Some Name_mode.normal) arg canonical ty
                  Hsatisfied Htype)).
  rewrite (agrees_eval_simple tenv valuation vars arg Hagrees);
    [ exact Heval | ].
  intros x ->. exact Hmem.
Qed.

(* ======================================================================== *)
(* The type of an argument, computed                                        *)
(* ======================================================================== *)

Lemma simplify_simple_answer (tenv : Typing_env.t) (s canonical : simple)
    (ty : flambda_type) :
  simplify_simple tenv s = Some (ty, canonical) ->
  argument_type tenv s ty ∧ canonical_in_terms tenv s canonical.
Proof.
  unfold simplify_simple.
  destruct (Typing_env.mem_simple (Some Name_mode.normal) tenv s) eqn:Hmem;
    [ | discriminate ].
  destruct (Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s)
    as [[[ty' canonical'] | ] | ] eqn:Htype; try discriminate.
  intros [= <- <-].
  split; (split; [ exact Hmem | eexists; exact Htype ]).
Qed.

Lemma simplify_simples_answers (tenv : Typing_env.t) (args : list simple) :
  ∀ answers : list (flambda_type * simple),
    simplify_simples tenv args = Some answers ->
    Forall2 (argument_type tenv) args (map fst answers)
    ∧ Forall2 (canonical_in_terms tenv) args (map snd answers).
Proof.
  induction args as [ | arg args IH]; intros answers; simpl.
  - intros [= <-]. split; apply Forall2_nil.
  - destruct (simplify_simple tenv arg) as [[ty canonical] | ] eqn:Harg;
      [ | discriminate ].
    destruct (simplify_simples tenv args) as [answers' | ];
      [ | discriminate ].
    intros [= <-].
    destruct (simplify_simple_answer tenv arg canonical ty Harg)
      as [Hty Hcanonical].
    destruct (IH answers' eq_refl) as [Htys Hcanonicals].
    split; apply Forall2_cons; assumption.
Qed.

Lemma well_formed_input_b_sound (tenv : Typing_env.t) (ty : flambda_type) :
  well_formed_input_b tenv ty = true -> well_formed_input tenv ty.
Proof.
  unfold well_formed_input_b. intros Hwf. apply andb_true_iff in Hwf.
  destruct Hwf as [Htenv Hty].
  split; [ exact (proj1 (types_well_formed_b_iff tenv) Htenv) | exact Hty ].
Qed.

(* ======================================================================== *)
(* The arguments, and the two conclusions                                   *)
(* ======================================================================== *)

Theorem let_prim_canonical : let_prim_canonical_statement.
Proof.
  intros tenv x declared p args args' body Hargs vars conts statics hp rd Hsat.
  apply let_equivalent_step_exec;
    [ | intros v hp' rd' _ _; apply equivalent_refl ].
  simpl.
  rewrite (canonical_in_terms_simples tenv statics vars args args' Hsat
             Hargs).
  reflexivity.
Qed.

Theorem let_constant : let_constant_statement.
Proof.
  intros tenv x declared d c body Hknown vars conts statics hp rd Hsat.
  apply let_refines_step_forward.
  - intros v hp' rd' Hd.
    destruct (Hknown statics hp hp' rd rd' vars v Hsat Hd) as (-> & -> & ->).
    reflexivity.
  - intros v hp' rd' _ _. apply refines_refl.
Qed.

Theorem let_invalid : let_invalid_statement.
Proof.
  intros tenv x declared d body Hnone vars conts statics hp rd Hsat.
  apply step_eq_equivalent.
  rewrite step_let_exec, (Hnone statics hp rd vars Hsat), step_invalid.
  reflexivity.
Qed.

(* ======================================================================== *)
(* The run of the machine and the value                                     *)
(* ======================================================================== *)

(* The two conclusions ask of the run of the defining expression on the
   heap ([exec_named]); what the answer of a prover says is about its
   value ([eval_named]). The lemmas of this section go from the one to
   the other. *)

(* A [Let] whose defining expression has no value on any heap, in the
   states that satisfy the environment, in the form in which the proofs
   have it. *)
Lemma let_no_value_exec (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (defining_expr : named) (body : expr) :
  (∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
     (vars : env) (r : value),
    env_satisfies_strictly tenv statics vars ->
    exec_named statics hp rd vars defining_expr = Some (r, hp', rd') ->
    False) ->
  expr_equivalent_under tenv (Let x declared defining_expr body) Invalid.
Proof.
  intros Hnone. apply let_invalid. intros statics hp rd vars Hsat.
  destruct (exec_named statics hp rd vars defining_expr)
    as [[[r hp'] rd'] | ] eqn:Hexec;
    [ exfalso; exact (Hnone statics hp hp' rd rd' vars r Hsat Hexec)
    | reflexivity ].
Qed.

(* Where a defining expression that reads no cell
   (FlambdaHeapSpec.named_reads_heap) runs on a heap, its value is the
   one that [eval_named] gives it, and the heap and the record are as
   they came. That is one way only: the run asks too that the locations
   among the arguments be bound in the heap, so one that has a value by
   [eval_named] may not run. The two conclusions with the premise on
   that value. *)
Lemma let_constant_pure (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (defining_expr : named) (c : const)
    (body : expr) :
  named_reads_heap defining_expr = false ->
  (∀ (statics : static_env) (vars : env) (v : value),
    env_satisfies_strictly tenv statics vars ->
    eval_named statics vars defining_expr = Some v ->
    v = Value_const c) ->
  expr_refines_under tenv (Let x declared defining_expr body)
    (Let x declared (Simple (Const c)) body).
Proof.
  intros Hpure Hknown. apply let_constant.
  intros statics hp hp' rd rd' vars v Hsat Hexec.
  destruct (exec_named_pure statics hp hp' rd rd' vars defining_expr v Hpure
              Hexec) as (Heval & -> & ->).
  split; [ exact (Hknown statics vars v Hsat Heval) | split; reflexivity ].
Qed.

Lemma let_no_value_pure (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (defining_expr : named) (body : expr) :
  named_reads_heap defining_expr = false ->
  (∀ (statics : static_env) (vars : env) (r : value),
    env_satisfies_strictly tenv statics vars ->
    eval_named statics vars defining_expr = Some r ->
    False) ->
  expr_equivalent_under tenv (Let x declared defining_expr body) Invalid.
Proof.
  intros Hpure Hnone. apply let_no_value_exec.
  intros statics hp hp' rd rd' vars r Hsat Hexec.
  exact (Hnone statics vars r Hsat
           (proj1 (exec_named_pure statics hp hp' rd rd' vars defining_expr r
                     Hpure Hexec))).
Qed.

(* A unary primitive that may read a cell (FlambdaHeapSpec.prim_reads_heap:
   a load from a block, [Get_tag], [Is_int], [Is_null]), where it runs,
   has done one of two things. On a value that is not a location its
   value is the one that [eval_prim] gives it. On a location it has
   read the cell that the heap has there
   (FlambdaHeapSpec.exec_unary_on_cell). Either way the heap is as it
   came. *)
(* The record is as it came too. Where the value is not a location that
   is [exec_named_no_location], which holds whatever the record comes to
   do. Where it is a location it is [exec_named_prim_some], which says so
   of every primitive today: that case is to be seen again when the
   record acts (decisions D49 and D52). *)
Lemma unary_prim_exec (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (p : unary_primitive) (arg : simple)
    (r : value) :
  exec_named statics hp rd vars (Prim (Unary p) [arg]) = Some (r, hp', rd') ->
  ∃ v : value,
    eval_simple vars arg = Some v
    ∧ hp' = hp
    ∧ rd' = rd
    ∧ ((∀ l : location, v <> Value_location l)
       ∧ eval_prim statics (Unary p) [v] = Some r
       ∨ ∃ (l : location) (c : cell),
           v = Value_location l
           ∧ lookup l hp = Some c
           ∧ exec_unary_on_cell p l c = Some r).
Proof.
  intros Hexec.
  destruct (proj1 (exec_named_prim_some statics hp hp' rd rd' vars (Unary p)
                     [arg] r) Hexec) as (vs & Hvs & Hprim & Hrd).
  simpl in Hvs.
  destruct (eval_simple vars arg) as [v | ] eqn:Hv; [ | discriminate ].
  injection Hvs as <-. exists v. split; [ reflexivity | ].
  destruct (is_location v) eqn:Hlocation.
  - destruct v; try discriminate Hlocation.
    apply exec_prim_bound in Hprim. destruct Hprim as [_ Hbound].
    simpl in Hbound.
    destruct (lookup l hp) as [c | ] eqn:Hcell; [ | discriminate ].
    simpl in Hbound.
    destruct (exec_unary_on_cell p l c) as [w | ] eqn:Hw; [ | discriminate ].
    injection Hbound as <- <-.
    split; [ reflexivity | ]. split; [ exact Hrd | ].
    right. exists l, c. split; [ reflexivity | ].
    split; [ exact Hcell | exact Hw ].
  - assert (Hnot : ∀ l : location, v <> Value_location l).
    { intros l ->. discriminate Hlocation. }
    rewrite (exec_named_no_location statics hp rd vars (Unary p) [arg] [v])
      in Hexec;
      [ | intros q; discriminate | simpl; rewrite Hv; reflexivity
        | intros l [Heq | []]; exact (Hnot l Heq) ].
    simpl in Hexec. rewrite Hv in Hexec. simpl in Hexec.
    destruct (eval_prim statics (Unary p) [v]) as [w | ] eqn:Hw;
      [ | discriminate ].
    injection Hexec as <- <- <-.
    split; [ reflexivity | ]. split; [ reflexivity | ].
    left. split; [ exact Hnot | reflexivity ].
Qed.

(* No head of what is not null ([head_of_kind_value_non_null]) has a
   location: [Mutable_block] has no value, and every other one asks for
   a tagged immediate, for a block that is a value or that a symbol is
   bound to, for a boxed number, for a closure or for a string. A type
   of kind value of which nothing is known, or nothing but that it is
   not null, does have one: that is the lemma after this one.
   (FlambdaTypeMeaningSpec.v says that with a heap [Mutable_block] is to
   mean a mutable block, and that such a block is to be one of [Variant]
   too. This lemma goes on that day, and what a prover answers of a
   location is to be read again: the lemmas below that end in
   [no_location] rest on it.) *)
Lemma location_no_head (statics : static_env) (vars : env) (l : location)
    (head : head_of_kind_value_non_null) :
  ~ head_of_kind_value_non_null_satisfied statics vars (Value_location l)
      head.
Proof.
  intros Hhead.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k' contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    autorewrite with type_satisfied in Hhead.
  - destruct Hhead as [[i [Heq _]] | [tag [fields [Hblock _]]]];
      discriminate.
  - exact Hhead.
  - destruct Hhead as [c [Hboxed _]]. discriminate.
  - destruct Hhead as [slot [functions [value_slots [Hclosure _]]]].
    discriminate.
  - destruct Hhead as [bytes [Hstring _]]. discriminate.
Qed.

(* So a type that a location has, in a state that satisfies the
   environment, is of kind value and, once its head is expanded, has
   nothing known of what is not null. This is of the expanded head, which
   is what the provers read: the type itself may be an alias of a name
   that holds the location. *)
Lemma location_expanded (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (t : flambda_type) (d : descr_oub) (l : location) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation (Value_location l) t ->
  Provers.expand_head tenv t = Some d ->
  d = Oub_value Unknown
  ∨ ∃ is_null, d = Oub_value (Ok (Head_of_kind_value Unknown is_null)).
Proof.
  intros Hsat Hv Hd.
  pose proof (expanded_kind statics valuation tenv t d _ Hsat Hv Hd) as Hkind.
  destruct d as [oub | oub | k oub | oub | oub]; simpl in Hkind;
    try discriminate Hkind.
  pose proof (expanded_value_cases statics valuation tenv t oub _ Hsat Hv Hd)
    as Hcases.
  destruct oub as [ | [non_null is_null] | ].
  - left. reflexivity.
  - destruct Hcases as [[Hnull _] | [_ Hnon_null]]; [ discriminate Hnull | ].
    destruct non_null as [ | head | ].
    + right. exists is_null. reflexivity.
    + exfalso. exact (location_no_head statics valuation l head Hnon_null).
    + destruct Hnon_null.
  - destruct Hcases.
Qed.

(* The three ways in which a prover reads a head of kind value answer
   nothing of such a type. *)
Lemma gen_value_to_meet_location {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (l : location) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation (Value_location l) t ->
  Provers.gen_value_to_meet prove_gen tenv t = Some Meet_shortcut.Need_meet
  ∨ Provers.gen_value_to_meet prove_gen tenv t = None.
Proof.
  intros Hsat Hv. unfold Provers.gen_value_to_meet.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | right; reflexivity ].
  destruct (location_expanded statics valuation tenv t d l Hsat Hv Hd)
    as [-> | [is_null ->]];
    left; reflexivity.
Qed.

Lemma gen_value_to_proof_location {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (l : location) :
  typing_env_satisfied_strictly statics valuation tenv ->
  type_satisfied statics valuation (Value_location l) t ->
  Provers.gen_value_to_proof prove_gen tenv t
  = Some Proof_of_property.Unknown
  ∨ Provers.gen_value_to_proof prove_gen tenv t = None.
Proof.
  intros Hsat Hv. unfold Provers.gen_value_to_proof.
  destruct (Provers.expand_head tenv t) as [d | ] eqn:Hd;
    [ | right; reflexivity ].
  destruct (location_expanded statics valuation tenv t d l Hsat Hv Hd)
    as [-> | [is_null ->]];
    [ | destruct is_null ]; left; reflexivity.
Qed.

(* No value of the type is a location, in a state that satisfies the
   environment. *)
Definition no_location (tenv : Typing_env.t) (ty : flambda_type) : Prop :=
  ∀ (statics : static_env) (valuation : env) (l : location),
    typing_env_satisfied_strictly statics valuation tenv ->
    ~ type_satisfied statics valuation (Value_location l) ty.

(* So it is where a prover that reads the head of the type answers
   something. *)
Lemma meet_answer_no_location {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (tenv : Typing_env.t) (ty : flambda_type) (answer : Meet_shortcut.t A) :
  Provers.gen_value_to_meet prove_gen tenv ty = Some answer ->
  answer <> Meet_shortcut.Need_meet ->
  no_location tenv ty.
Proof.
  intros Hanswer Hnot statics valuation l Hsat Hv.
  destruct (gen_value_to_meet_location prove_gen statics valuation tenv ty l
              Hsat Hv) as [Heq | Heq];
    rewrite Heq in Hanswer;
    [ injection Hanswer as <-; exact (Hnot eq_refl) | discriminate ].
Qed.

Lemma proof_answer_no_location {A : Type}
    (prove_gen :
      Typing_env.t -> head_of_kind_value_non_null
      -> option (Generic_proof.t A))
    (tenv : Typing_env.t) (ty : flambda_type) (x : A) :
  Provers.gen_value_to_proof prove_gen tenv ty
  = Some (Proof_of_property.Proved x) ->
  no_location tenv ty.
Proof.
  intros Hanswer statics valuation l Hsat Hv.
  destruct (gen_value_to_proof_location prove_gen statics valuation tenv ty l
              Hsat Hv) as [Heq | Heq];
    rewrite Heq in Hanswer; discriminate.
Qed.

(* A unary primitive on an argument whose type has no location: where
   it runs, its value is the one that [eval_named] gives it, and the
   heap and the record are as they came. *)
Lemma unary_prim_exec_typed (tenv : Typing_env.t) (statics : static_env)
    (hp hp' : heap) (rd rd' : read_record) (vars : env)
    (p : unary_primitive) (arg : simple) (ty : flambda_type) (r : value) :
  env_satisfies_strictly tenv statics vars ->
  argument_type tenv arg ty ->
  no_location tenv ty ->
  exec_named statics hp rd vars (Prim (Unary p) [arg]) = Some (r, hp', rd') ->
  eval_named statics vars (Prim (Unary p) [arg]) = Some r
  ∧ hp' = hp
  ∧ rd' = rd.
Proof.
  intros Hsat Hty Hno Hexec.
  destruct (unary_prim_exec statics hp hp' rd rd' vars p arg r Hexec)
    as (v & Hv & -> & -> & [[_ Hprim] | (l & c & -> & _)]).
  - split; [ | split; reflexivity ].
    rewrite eval_named_prim. simpl. rewrite Hv. exact Hprim.
  - exfalso.
    destruct (argument_type_satisfied tenv statics vars arg ty _ Hsat Hty Hv)
      as [valuation [Hsatisfied [_ Htyped]]].
    exact (Hno statics valuation l Hsatisfied Htyped).
Qed.

Lemma let_constant_unary (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (p : unary_primitive) (arg : simple)
    (ty : flambda_type) (c : const) (body : expr) :
  argument_type tenv arg ty ->
  no_location tenv ty ->
  (∀ (statics : static_env) (vars : env) (v : value),
    env_satisfies_strictly tenv statics vars ->
    eval_named statics vars (Prim (Unary p) [arg]) = Some v ->
    v = Value_const c) ->
  expr_refines_under tenv (Let x declared (Prim (Unary p) [arg]) body)
    (Let x declared (Simple (Const c)) body).
Proof.
  intros Hty Hno Hknown. apply let_constant.
  intros statics hp hp' rd rd' vars v Hsat Hexec.
  destruct (unary_prim_exec_typed tenv statics hp hp' rd rd' vars p arg ty v
              Hsat Hty Hno Hexec) as (Hnamed & -> & ->).
  split; [ exact (Hknown statics vars v Hsat Hnamed) | split; reflexivity ].
Qed.

Lemma let_no_value_unary (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (p : unary_primitive) (arg : simple)
    (ty : flambda_type) (body : expr) :
  argument_type tenv arg ty ->
  no_location tenv ty ->
  (∀ (statics : static_env) (vars : env) (r : value),
    env_satisfies_strictly tenv statics vars ->
    eval_named statics vars (Prim (Unary p) [arg]) = Some r ->
    False) ->
  expr_equivalent_under tenv (Let x declared (Prim (Unary p) [arg]) body)
    Invalid.
Proof.
  intros Hty Hno Hnone. apply let_no_value_exec.
  intros statics hp hp' rd rd' vars r Hsat Hexec.
  exact (Hnone statics vars r Hsat
           (proj1 (unary_prim_exec_typed tenv statics hp hp' rd rd' vars p arg
                     ty r Hsat Hty Hno Hexec))).
Qed.

(* The values of arguments that have values. *)
Lemma eval_simples_values (vars : env) (args : list simple) :
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    Forall2 (fun arg v => eval_simple vars arg = Some v) args vs.
Proof.
  induction args as [ | arg args IH]; intros vs; simpl.
  - intros [= <-]. apply Forall2_nil.
  - destruct (eval_simple vars arg) as [v | ] eqn:Hv; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ]; [ | discriminate ].
    intros [= <-]. apply Forall2_cons; [ exact Hv | exact (IH vs' eq_refl) ].
Qed.

Theorem let_prim_kind_mismatch : let_prim_kind_mismatch_statement.
Proof.
  intros tenv x declared p args tys body Hnot Htys Hkinds.
  apply let_no_value_exec. intros statics hp hp' rd rd' vars r Hsat Hnamed.
  destruct (proj1 (exec_named_prim_some statics hp hp' rd rd' vars p args r)
              Hnamed) as (vs & Hvs & Hprim & _).
  destruct (exec_prim_kinds statics hp hp' p vs r Hnot Hprim) as [Hwell _].
  assert (Hsame :
    map kind_of_value vs = map FlambdaTypeGrammar.kind tys).
  { pose proof (eval_simples_values vars args vs Hvs) as Hvalues.
    clear Hvs Hnamed Hprim Hwell Hkinds. revert vs Hvalues.
    induction Htys as [ | arg ty args tys Hty Htys IH]; intros vs Hvalues;
      inversion Hvalues as [ | arg' v args' vs' Hv Hrest]; subst;
      [ reflexivity | ].
    simpl. f_equal; [ | exact (IH vs' Hrest) ].
    destruct (argument_type_satisfied tenv statics vars arg ty v Hsat Hty
                Hv) as [valuation [_ [_ Htyped]]].
    exact (type_satisfied_kind statics valuation v ty Htyped). }
  rewrite Hsame in Hwell. congruence.
Qed.

(* ======================================================================== *)
(* A primitive applied in a state that satisfies the environment            *)
(* ======================================================================== *)

(* A unary primitive that has a value was applied to the value of its
   argument. *)
Lemma unary_prim_value (statics : static_env) (vars : env)
    (p : unary_primitive) (arg : simple) (r : value) :
  eval_named statics vars (Prim (Unary p) [arg]) = Some r ->
  ∃ v : value,
    eval_simple vars arg = Some v ∧ eval_prim statics (Unary p) [v] = Some r.
Proof.
  simpl. destruct (eval_simple vars arg) as [v | ]; [ | discriminate ].
  intros Hprim. exists v. split; [ reflexivity | exact Hprim ].
Qed.

(* So to the value of an argument that has the type of the argument. *)
Lemma unary_prim_typed (tenv : Typing_env.t) (statics : static_env)
    (vars : env) (p : unary_primitive) (arg : simple) (ty : flambda_type)
    (r : value) :
  env_satisfies_strictly tenv statics vars ->
  argument_type tenv arg ty ->
  eval_named statics vars (Prim (Unary p) [arg]) = Some r ->
  ∃ (valuation : env) (v : value),
    typing_env_satisfied_strictly statics valuation tenv
    ∧ type_satisfied statics valuation v ty
    ∧ eval_prim statics (Unary p) [v] = Some r.
Proof.
  intros Hsat Hty Hnamed.
  destruct (unary_prim_value statics vars p arg r Hnamed) as [v [Hv Hprim]].
  destruct (argument_type_satisfied tenv statics vars arg ty v Hsat Hty Hv)
    as [valuation [Hsatisfied [_ Htyped]]].
  exists valuation, v.
  split; [ exact Hsatisfied | ]. split; [ exact Htyped | exact Hprim ].
Qed.

(* Two answers about one value are the same. *)
Lemma is_int_is_unique (b b' : bool) (v : value) :
  is_int_is b v -> is_int_is b' v -> b = b'.
Proof. destruct b, b'; simpl; intros H H'; try reflexivity; contradiction. Qed.

Lemma is_null_is_unique (b b' : bool) (v : value) :
  is_null_is b v -> is_null_is b' v -> b = b'.
Proof. destruct b, b'; simpl; intros H H'; try reflexivity; contradiction. Qed.

Lemma naked_bool_const (b : bool) :
  naked_bool b = Value_const (Naked_immediate (if b then 1 else 0)).
Proof. reflexivity. Qed.

(* ======================================================================== *)
(* [Is_int] and [Get_tag], through [Provers.prove_is_int]                   *)
(* ======================================================================== *)

Theorem let_is_int_proved : let_is_int_proved_statement.
Proof.
  intros tenv x declared arg ty b body Hty Hprove.
  apply (let_constant_unary tenv x declared _ arg ty _ body Hty);
    [ exact (proof_answer_no_location _ tenv ty b Hprove) | ].
  intros statics vars r Hsat Hnamed.
  destruct (unary_prim_typed tenv statics vars _ arg ty r Hsat Hty Hnamed)
    as [valuation [v [Hsatisfied [Htyped Hprim]]]].
  rewrite <- naked_bool_const.
  exact (prove_is_int_is_int statics valuation tenv ty b v r
           Hsatisfied Htyped Hprove Hprim).
Qed.

Theorem let_get_tag_of_immediate : let_get_tag_of_immediate_statement.
Proof.
  intros tenv x declared arg ty body Hty Hprove.
  apply (let_no_value_unary tenv x declared _ arg ty body Hty);
    [ exact (proof_answer_no_location _ tenv ty true Hprove) | ].
  intros statics vars r Hsat Hnamed.
  destruct (unary_prim_typed tenv statics vars _ arg ty r Hsat Hty Hnamed)
    as [valuation [v [Hsatisfied [Htyped Hprim]]]].
  rewrite (prove_is_int_get_tag statics valuation tenv ty v
             Hsatisfied Htyped Hprove) in Hprim.
  discriminate.
Qed.

(* ======================================================================== *)
(* The relational primitives                                                *)
(* ======================================================================== *)

Theorem is_int_relation_type : is_int_relation_type_statement.
Proof.
  intros statics valuation scrutinee w r Hw Hprim.
  destruct (is_int_defined statics true w r Hprim)
    as [Hvariants [b [-> Hb]]].
  apply is_int_for_scrutinee_satisfied.
  exists (if b then 1 else 0), w.
  split; [ apply naked_bool_const | ]. split; [ exact Hw | ].
  destruct b; simpl in Hb.
  - left. split; [ exact Hb | reflexivity ].
  - right. split; [ | reflexivity ].
    destruct (Hvariants eq_refl) as [Himmediate | Hblock];
      [ contradiction | exact Hblock ].
Qed.

Theorem is_null_relation_type : is_null_relation_type_statement.
Proof.
  intros statics valuation scrutinee w r Hw Hprim.
  destruct (is_null_defined statics w r Hprim) as [Hkind [b [-> Hb]]].
  apply is_null_for_scrutinee_satisfied.
  exists (if b then 1 else 0), w.
  split; [ apply naked_bool_const | ]. split; [ exact Hw | ].
  destruct b; simpl in Hb.
  - left. split; [ exact Hb | reflexivity ].
  - right. split; [ exact Hkind | ]. split; [ exact Hb | reflexivity ].
Qed.

Theorem get_tag_relation_type : get_tag_relation_type_statement.
Proof.
  intros statics valuation block w r Hw Hprim Htag.
  destruct (get_tag_defined statics w r Hprim) as [tag [fields [Hblock ->]]].
  apply get_tag_for_block_satisfied.
  exists tag, w. split; [ reflexivity | ]. split; [ exact Hw | ].
  right. split; [ exists fields; exact Hblock | ].
  exact (Htag tag fields Hblock).
Qed.

Theorem let_is_int_variant_only : let_is_int_variant_only_statement.
Proof.
  intros tenv x declared arg ty body Hty.
  (* What the answer says of the value of the argument. *)
  assert (Hsound :
    ∀ (statics : static_env) (vars : env) (r : value),
      env_satisfies_strictly tenv statics vars ->
      eval_named statics vars (Prim (Unary (Is_int true)) [arg]) = Some r ->
      match Provers.meet_is_int_variant_only tenv ty with
      | Some (Meet_shortcut.Known_result b) => r = naked_bool b
      | Some Meet_shortcut.Invalid => False
      | Some Meet_shortcut.Need_meet | None => True
      end).
  { intros statics vars r Hsat Hnamed.
    destruct (unary_prim_typed tenv statics vars _ arg ty r Hsat Hty Hnamed)
      as [valuation [v [Hsatisfied [Htyped Hprim]]]].
    destruct (is_int_defined statics true v r Hprim)
      as [Hvariants [b' [-> Hb']]].
    pose proof (meet_is_int_variant_only_sound statics valuation tenv
                  ty v Hsatisfied I Htyped (Hvariants eq_refl)) as Hanswer.
    destruct (Provers.meet_is_int_variant_only tenv ty) as [[b | | ] | ];
      try exact I; [ | exact Hanswer ].
    rewrite (is_int_is_unique b b' v Hanswer Hb'). reflexivity. }
  destruct (Provers.meet_is_int_variant_only tenv ty) as [[b | | ] | ]
    eqn:Hanswer;
    try exact I.
  - apply (let_constant_unary tenv x declared _ arg ty _ body Hty);
      [ exact (meet_answer_no_location _ tenv ty _ Hanswer
                 ltac:(discriminate))
      | ].
    intros statics vars r Hsat Hnamed.
    rewrite <- naked_bool_const. exact (Hsound statics vars r Hsat Hnamed).
  - apply (let_no_value_unary tenv x declared _ arg ty body Hty);
      [ exact (meet_answer_no_location _ tenv ty _ Hanswer
                 ltac:(discriminate))
      | exact Hsound ].
Qed.

Theorem let_get_tag_proved : let_get_tag_proved_statement.
Proof.
  intros tenv x declared arg ty tags tag body Hty Hprove Htags.
  apply (let_constant_unary tenv x declared _ arg ty _ body Hty);
    [ exact (proof_answer_no_location _ tenv ty tags Hprove) | ].
  intros statics vars r Hsat Hnamed.
  destruct (unary_prim_typed tenv statics vars _ arg ty r Hsat Hty Hnamed)
    as [valuation [v [Hsatisfied [Htyped Hprim]]]].
  destruct (get_tag_defined statics v r Hprim)
    as [tag' [fields' [Hblock' ->]]].
  destruct (prove_get_tag_sound statics valuation tenv ty tags v
              Hsatisfied I Htyped Hprove)
    as [tag'' [fields'' [Hblock'' Hin]]].
  rewrite Hblock' in Hblock''. injection Hblock'' as -> _.
  rewrite (Htags tag'' Hin). reflexivity.
Qed.

Theorem let_is_null : let_is_null_statement.
Proof.
  intros tenv x declared arg ty body Hty.
  assert (Hsound :
    ∀ (statics : static_env) (vars : env) (r : value),
      env_satisfies_strictly tenv statics vars ->
      eval_named statics vars (Prim (Unary Is_null) [arg]) = Some r ->
      match Provers.meet_is_null tenv ty with
      | Some (Meet_shortcut.Known_result b) => r = naked_bool b
      | Some Meet_shortcut.Invalid => False
      | Some Meet_shortcut.Need_meet | None => True
      end).
  { intros statics vars r Hsat Hnamed.
    destruct (unary_prim_typed tenv statics vars _ arg ty r Hsat Hty Hnamed)
      as [valuation [v [Hsatisfied [Htyped Hprim]]]].
    destruct (is_null_defined statics v r Hprim) as [Hkind [b' [-> Hb']]].
    pose proof (meet_is_null_sound statics valuation tenv ty v
                  Hsatisfied I Htyped Hkind) as Hanswer.
    destruct (Provers.meet_is_null tenv ty) as [[b | | ] | ];
      try exact I; [ | exact Hanswer ].
    rewrite (is_null_is_unique b b' v Hanswer Hb'). reflexivity. }
  (* The argument may be a location: the prover answers that a value
     is not null from a head of which no more is known. On a location
     the primitive answers no, as the prover does. *)
  assert (Hlocation :
    ∀ (statics : static_env) (vars : env) (l : location),
      env_satisfies_strictly tenv statics vars ->
      eval_simple vars arg = Some (Value_location l) ->
      match Provers.meet_is_null tenv ty with
      | Some (Meet_shortcut.Known_result b) => b = false
      | Some Meet_shortcut.Invalid => False
      | Some Meet_shortcut.Need_meet | None => True
      end).
  { intros statics vars l Hsat Hv.
    destruct (argument_type_satisfied tenv statics vars arg ty _ Hsat Hty Hv)
      as [valuation [Hsatisfied [_ Htyped]]].
    pose proof (meet_is_null_sound statics valuation tenv ty _
                  Hsatisfied I Htyped eq_refl) as Hanswer.
    destruct (Provers.meet_is_null tenv ty) as [[b | | ] | ];
      try exact I; [ | exact Hanswer ].
    destruct b; [ discriminate Hanswer | reflexivity ]. }
  assert (Hcell :
    ∀ (l : location) (c : cell) (r : value),
      exec_unary_on_cell Is_null l c = Some r -> r = naked_bool false).
  { intros l c r. destruct c; simpl; intros [= <-]; reflexivity. }
  destruct (Provers.meet_is_null tenv ty) as [[b | | ] | ]; try exact I.
  - apply let_constant. intros statics hp hp' rd rd' vars r Hsat Hexec.
    destruct (unary_prim_exec statics hp hp' rd rd' vars _ arg r Hexec)
      as (v & Hv & -> & -> & [[_ Hprim] | (l & c & -> & _ & Hr)]);
      (split; [ | split; reflexivity ]); rewrite <- naked_bool_const.
    + apply (Hsound statics vars r Hsat).
      rewrite eval_named_prim. simpl. rewrite Hv. exact Hprim.
    + rewrite (Hcell l c r Hr), (Hlocation statics vars l Hsat Hv).
      reflexivity.
  - apply let_no_value_exec. intros statics hp hp' rd rd' vars r Hsat Hexec.
    destruct (unary_prim_exec statics hp hp' rd rd' vars _ arg r Hexec)
      as (v & Hv & _ & _ & [[_ Hprim] | (l & c & -> & _ & _)]).
    + apply (Hsound statics vars r Hsat).
      rewrite eval_named_prim. simpl. rewrite Hv. exact Hprim.
    + exact (Hlocation statics vars l Hsat Hv).
Qed.

(* ======================================================================== *)
(* A comparison of tagged immediates                                        *)
(* ======================================================================== *)

(* A comparison of tagged immediates that has a value was applied to
   two tagged immediates, and its value is the comparison of their
   numbers. *)
Lemma int_comp_tagged_defined (statics : static_env)
    (behaviour : int_comparison_behaviour) (v1 v2 r : value) :
  eval_prim statics (Binary (Int_comp Standard_tagged_immediate behaviour))
    [v1; v2]
  = Some r ->
  ∃ i1 i2 : Z,
    v1 = Value_const (Tagged_immediate i1)
    ∧ v2 = Value_const (Tagged_immediate i2)
    ∧ r
      = Value_const
          (Naked_immediate
             (eval_int_comp (int_width Standard_tagged_immediate) behaviour
                i1 i2)).
Proof.
  cbn [eval_prim eval_binary_primitive].
  destruct (int_of_value Standard_tagged_immediate v1) as [a | ] eqn:Ha;
    [ | discriminate ].
  destruct (int_of_value Standard_tagged_immediate v2) as [b | ] eqn:Hb;
    [ | discriminate ].
  intros [= <-].
  assert (Hint :
    ∀ (v : value) (i : Z),
      int_of_value Standard_tagged_immediate v = Some i ->
      v = Value_const (Tagged_immediate i)).
  { intros v i Hv. exact (proj1 (int_of_value_const _ _ _ Hv)). }
  exists a, b.
  split; [ exact (Hint v1 a Ha) | ]. split; [ exact (Hint v2 b Hb) | ].
  reflexivity.
Qed.

Theorem let_int_comp_tagged : let_int_comp_tagged_statement.
Proof.
  intros tenv x declared behaviour arg1 arg2 ty1 ty2 body Hty1 Hty2
    Hwf1 Hwf2 comparison.
  (* What the answers say of the values of the arguments. *)
  assert (Hsound :
    ∀ (statics : static_env) (vars : env) (r : value),
      env_satisfies_strictly tenv statics vars ->
      eval_named statics vars comparison = Some r ->
      ∃ i1 i2 : Z,
        r
        = Value_const
            (Naked_immediate
               (eval_int_comp (int_width Standard_tagged_immediate)
                  behaviour i1 i2))
        ∧ match Provers.meet_equals_tagged_immediates tenv ty1 with
          | Some (Meet_shortcut.Known_result imms1) => i1 ∈ imms1
          | Some Meet_shortcut.Invalid => False
          | Some Meet_shortcut.Need_meet | None => True
          end
        ∧ match Provers.meet_equals_tagged_immediates tenv ty2 with
          | Some (Meet_shortcut.Known_result imms2) => i2 ∈ imms2
          | Some Meet_shortcut.Invalid => False
          | Some Meet_shortcut.Need_meet | None => True
          end).
  { intros statics vars r Hsat Hnamed. unfold comparison in Hnamed.
    simpl in Hnamed.
    destruct (eval_simple vars arg1) as [v1 | ] eqn:Hv1; [ | discriminate ].
    destruct (eval_simple vars arg2) as [v2 | ] eqn:Hv2; [ | discriminate ].
    change (eval_prim statics
              (Binary (Int_comp Standard_tagged_immediate behaviour))
              [v1; v2]
            = Some r) in Hnamed.
    destruct (int_comp_tagged_defined statics behaviour v1 v2 r Hnamed)
      as [i1 [i2 [-> [-> ->]]]].
    exists i1, i2. split; [ reflexivity | ].
    (* The answer about one argument. *)
    assert (Hone :
      ∀ (arg : simple) (ty : flambda_type) (i : Z),
        argument_type tenv arg ty ->
        well_formed_input tenv ty ->
        eval_simple vars arg = Some (Value_const (Tagged_immediate i)) ->
        match Provers.meet_equals_tagged_immediates tenv ty with
        | Some (Meet_shortcut.Known_result imms) => i ∈ imms
        | Some Meet_shortcut.Invalid => False
        | Some Meet_shortcut.Need_meet | None => True
        end).
    { intros arg ty i Hty Hwf Hv.
      destruct (argument_type_satisfied tenv statics vars arg ty _ Hsat Hty
                  Hv) as [valuation [Hsatisfied [_ Htyped]]].
      pose proof (meet_equals_tagged_immediates_sound statics
                    valuation tenv ty _ Hsatisfied Hwf Htyped
                    (ex_intro _ i eq_refl)) as Hanswer.
      destruct (Provers.meet_equals_tagged_immediates tenv ty)
        as [[imms | | ] | ]; try exact I; [ | exact Hanswer ].
      destruct Hanswer as [j [[= <-] Hin]]. exact Hin. }
    split;
      [ exact (Hone arg1 ty1 i1 Hty1 Hwf1 Hv1)
      | exact (Hone arg2 ty2 i2 Hty2 Hwf2 Hv2) ]. }
  assert (Hinvalid :
    (∀ (statics : static_env) (vars : env) (r : value),
      env_satisfies_strictly tenv statics vars ->
      eval_named statics vars comparison = Some r ->
      False) ->
    expr_equivalent_under tenv (Let x declared comparison body) Invalid).
  { intros Hnone. apply let_no_value_pure; [ reflexivity | exact Hnone ]. }
  destruct (Provers.meet_equals_tagged_immediates tenv ty1)
    as [[imms1 | | ] | ],
    (Provers.meet_equals_tagged_immediates tenv ty2) as [[imms2 | | ] | ];
    try exact I;
    try (apply Hinvalid; intros statics vars r Hsat Hnamed;
         destruct (Hsound statics vars r Hsat Hnamed)
           as [i1 [i2 [_ [H1 H2]]]];
         first [ exact H1 | exact H2 ]).
  intros result Hresult.
  apply let_constant_pure; [ reflexivity | ].
  intros statics vars r Hsat Hnamed.
  destruct (Hsound statics vars r Hsat Hnamed) as [i1 [i2 [-> [H1 H2]]]].
  rewrite (Hresult i1 i2 H1 H2). reflexivity.
Qed.

(* ======================================================================== *)
(* Physical equality                                                        *)
(* ======================================================================== *)

(* The value of an argument has the type of the argument under any
   valuation that satisfies the environment and agrees with the
   machine: the two arguments of a primitive under the same one. *)
Lemma argument_type_in_valuation (tenv : Typing_env.t)
    (statics : static_env) (valuation vars : env) (arg : simple)
    (ty : flambda_type) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  agrees_in_terms tenv valuation vars ->
  argument_type tenv arg ty ->
  eval_simple vars arg = Some v ->
  type_satisfied statics valuation v ty.
Proof.
  intros Hsatisfied Hagrees [Hmem [canonical Htype]] Heval.
  apply (proj2 (type_simple_in_term_satisfied (fun _ => False) statics
                  valuation tenv (Some Name_mode.normal) arg canonical ty
                  Hsatisfied Htype)).
  rewrite (agrees_eval_simple tenv valuation vars arg Hagrees);
    [ exact Heval | ].
  intros x ->. exact Hmem.
Qed.

(* [Phys_equal] where one of the two values is a location
   (FlambdaHeapSpec.eval_phys_equal_location) answers yes exactly of one
   location against itself. *)
Lemma eval_phys_equal_location_same (statics : static_env) (v1 v2 : value)
    (same : bool) :
  eval_phys_equal_location statics v1 v2 = Some same ->
  same = true <-> v1 = v2.
Proof.
  destruct v1, v2; simpl; try discriminate;
    try (match goal with
         | |- (if ?test then _ else _) = _ -> _ => destruct test
         end;
         [ | discriminate ]);
    try (intros [= <-]; split; intros Heq; discriminate Heq).
  intros [= <-]. split.
  - intros Heq. apply Pos.eqb_eq in Heq. subst. reflexivity.
  - intros [= ->]. apply Pos.eqb_refl.
Qed.

(* What [Phys_equal] does on a heap: where neither value is a location,
   what [eval_prim] has; where one is, the comparison above. The heap is
   as it came, and so is the record ([exec_named_prim_some]: to be seen
   again when the record acts, decisions D49 and D52). *)
Lemma phys_equal_exec (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (op : equality_comparison)
    (arg1 arg2 : simple) (r : value) :
  exec_named statics hp rd vars (Prim (Binary (Phys_equal op)) [arg1; arg2])
  = Some (r, hp', rd') ->
  ∃ v1 v2 : value,
    eval_simple vars arg1 = Some v1
    ∧ eval_simple vars arg2 = Some v2
    ∧ hp' = hp
    ∧ rd' = rd
    ∧ (eval_prim statics (Binary (Phys_equal op)) [v1; v2] = Some r
       ∨ ∃ same : bool,
           eval_phys_equal_location statics v1 v2 = Some same
           ∧ r
             = naked_bool
                 match op with
                 | Phys_eq => same
                 | Phys_neq => negb same
                 end).
Proof.
  intros Hexec.
  destruct (proj1 (exec_named_prim_some statics hp hp' rd rd' vars
                     (Binary (Phys_equal op)) [arg1; arg2] r) Hexec)
    as (vs & Hvs & Hprim & Hrd).
  simpl in Hvs.
  destruct (eval_simple vars arg1) as [v1 | ] eqn:Hv1; [ | discriminate ].
  destruct (eval_simple vars arg2) as [v2 | ] eqn:Hv2; [ | discriminate ].
  injection Hvs as <-. exists v1, v2.
  split; [ reflexivity | ]. split; [ reflexivity | ].
  apply exec_prim_bound in Hprim. destruct Hprim as [_ Hbound].
  (* The definition looks at the first value before it looks at the
     primitive; whatever that value is, it does this. *)
  assert (Hcases :
    (if is_location v1 || is_location v2
     then
       let* same := eval_phys_equal_location statics v1 v2 in
       Some
         (naked_bool
            match op with
            | Phys_eq => same
            | Phys_neq => negb same
            end,
          hp)
     else
       let* w := eval_prim statics (Binary (Phys_equal op)) [v1; v2] in
       Some (w, hp))
    = Some (r, hp')).
  { destruct v1; exact Hbound. }
  clear Hbound. rename Hcases into Hbound.
  destruct (is_location v1 || is_location v2).
  - destruct (eval_phys_equal_location statics v1 v2) as [same | ] eqn:Hsame;
      simpl in Hbound; [ | discriminate Hbound ].
    injection Hbound as <- <-.
    split; [ reflexivity | ]. split; [ exact Hrd | ].
    right. exists same. split; reflexivity.
  - destruct (eval_prim statics (Binary (Phys_equal op)) [v1; v2])
      as [w | ] eqn:Hw; simpl in Hbound; [ | discriminate Hbound ].
    injection Hbound as <- <-.
    split; [ reflexivity | ]. split; [ exact Hrd | ].
    left. reflexivity.
Qed.

Theorem let_phys_equal : let_phys_equal_statement.
Proof.
  intros tenv x declared op arg1 arg2 ty1 ty2 b body Hty1 Hty2 Hwf1 Hwf2
    Hprove result.
  apply let_constant.
  intros statics hp hp' rd rd' vars r [_ [valuation [Hsatisfied Hagrees]]]
    Hexec.
  destruct (phys_equal_exec statics hp hp' rd rd' vars op arg1 arg2 r Hexec)
    as (v1 & v2 & Hv1 & Hv2 & -> & -> & Hvalue).
  split; [ | split; reflexivity ].
  pose proof (argument_type_in_valuation tenv statics valuation vars arg1 ty1
                v1 Hsatisfied Hagrees Hty1 Hv1) as Htyped1.
  pose proof (argument_type_in_valuation tenv statics valuation vars arg2 ty2
                v2 Hsatisfied Hagrees Hty2 Hv2) as Htyped2.
  rewrite <- naked_bool_const.
  destruct Hvalue as [Hprim | (same & Hsame & ->)].
  - exact (prove_physical_equality_phys_equal statics valuation tenv ty1 ty2
             b op v1 v2 r Hsatisfied Hwf1 Hwf2 Htyped1 Htyped2 Hprove Hprim).
  - (* One of the two is a location: the prover has said whether the two
       are one value, and that is what the comparison answers. *)
    pose proof (prove_physical_equality_sound statics valuation tenv ty1 ty2
                  b v1 v2 Hsatisfied Hwf1 Hwf2 Htyped1 Htyped2 Hprove) as Hb.
    destruct (eval_phys_equal_location_same statics v1 v2 same Hsame)
      as [Hto Hfrom].
    assert (Heq : same = b).
    { destruct b, same; try reflexivity.
      - exact (Hfrom Hb).
      - exfalso. exact (Hb (Hto eq_refl)). }
    rewrite Heq. reflexivity.
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* What the environment answers for an argument is computed. *)
Ltac argument_computed :=
  split; [ vm_compute; reflexivity | eexists; vm_compute; reflexivity ].

(* The first [Let] of the two examples: [x] is made equal to the
   constant. *)
Ltac example_first_let :=
  apply expr_refines_under_create;
  eapply (let_simple_refines_under Typing_env.create example_x Value
            (Const (Tagged_immediate 5)) _ _ Value);
  [ reflexivity | reflexivity | intros ? [=] | reflexivity
  | vm_compute; reflexivity | ].

(* Their end: [i] is made equal to 1, and the [Switch] is its arm. *)
Ltac example_switch_arm :=
  eapply (let_simple_refines_under _ example_i
            (Naked_number Kind_naked_immediate) (Const (Naked_immediate 1))
            _ _ (Naked_number Kind_naked_immediate));
  [ vm_compute; reflexivity | vm_compute; reflexivity | intros ? [=]
  | reflexivity | vm_compute; reflexivity
  | apply expr_equivalent_under_refines;
    apply (switch_known _ (Var example_i) 1);
    [ canonical_computed | reflexivity ] ].

Theorem example_is_int : example_is_int_statement.
Proof.
  intros k0 k1. example_first_let.
  eapply expr_refines_under_trans.
  - eapply (let_is_int_proved _ example_i _ (Var example_x) _ true);
      [ argument_computed | vm_compute; reflexivity ].
  - example_switch_arm.
Qed.

(* The types of an environment that is computed are well formed. *)
Ltac well_formed_input_computed :=
  split; [ | vm_compute; reflexivity ];
  apply (proj1 (types_well_formed_b_iff _)); vm_compute; reflexivity.

Theorem example_int_comp : example_int_comp_statement.
Proof.
  intros k0 k1. example_first_let.
  eapply expr_refines_under_trans.
  - match goal with
    | |- expr_refines_under ?tenv (Let _ _ _ ?body) _ =>
      pose proof (let_int_comp_tagged tenv example_i
                    (Naked_number Kind_naked_immediate)
                    (Yielding_bool (Int_lt Signed)) (Var example_x)
                    (Const (Tagged_immediate 7))
                    (alias_type_of Value (Const (Tagged_immediate 5)))
                    (alias_type_of Value (Const (Tagged_immediate 7))) body)
        as Hcomparison
    end.
    cbv zeta in Hcomparison.
    repeat match type of Hcomparison with
           | context [Provers.meet_equals_tagged_immediates ?tenv ?ty] =>
             let answer :=
               eval vm_compute in
                 (Provers.meet_equals_tagged_immediates tenv ty)
             in
             replace (Provers.meet_equals_tagged_immediates tenv ty)
               with answer in Hcomparison by (vm_compute; reflexivity)
           end.
    cbv beta iota in Hcomparison.
    refine (Hcomparison _ _ _ _ 1 _);
      [ argument_computed | argument_computed | well_formed_input_computed
      | well_formed_input_computed | ].
    intros i1 i2 [<- | []] [<- | []]. reflexivity.
  - example_switch_arm.
Qed.
