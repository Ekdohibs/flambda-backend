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


(* Proofs of the statements of FlambdaSimplifyPassSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaHeapSpec
  FlambdaHeapProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaCompareSpec FlambdaCompareProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaContextWhereSpec FlambdaContextWhereProofs FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaUnusedContsSpec FlambdaUnusedContsProofs
  FlambdaContAlikeProofs FlambdaSinkContsSpec FlambdaSinkContsProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs
  FlambdaTypeCreatorsSpec FlambdaTypeCreatorsProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvOperationsSpec FlambdaTypingEnvOperationsProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs
  FlambdaTypingEnvRefinementSpec FlambdaTypingEnvRefinementProofs
  FlambdaProvers FlambdaProversSpec FlambdaProversProofs
  FlambdaSimplifyPrimitivesSpec FlambdaSimplifyPrimitivesProofs
  FlambdaSimplifyProjectionsSpec FlambdaSimplifyProjectionsProofs
  FlambdaSimplifyBoxingSpec FlambdaSimplifyBoxingProofs
  FlambdaSimplifyArithSpec FlambdaSimplifyArithProofs
  FlambdaSimplifyApplySpec FlambdaSimplifyApplyProofs
  FlambdaSimplifyValueSlotsSpec FlambdaSimplifyValueSlotsProofs
  FlambdaSimplifyStaticsSpec FlambdaSimplifyStaticsProofs
  FlambdaSimplifyCodeEnvSpec FlambdaSimplifyCodeEnvProofs
  FlambdaSimplifyHandlersSpec FlambdaSimplifyHandlersProofs
  FlambdaSimplifyPassSpec FlambdaSimplifyUsesSpec FlambdaSimplifyUsesProofs.
Import ListNotations.

Local Open Scope Z_scope.

(* [code_env_in] is the function of FlambdaSimplifyCodeEnvSpec.v. The
   theorem of that name in FlambdaSimplifyCodeEnvProofs.v is not used
   here. *)
Local Abbreviation code_env_in :=
  FlambdaSimplifyCodeEnvSpec.code_env_in.

(* ======================================================================== *)
(* What an outcome says                                                     *)
(* ======================================================================== *)

Definition outcome_sound (tenv : Typing_env.t) (d : named) (o : outcome)
    : Prop :=
  match o with
  | Becomes_invalid =>
    ∀ (x : variable) (declared : flambda_kind) (body : expr),
      expr_equivalent_under tenv (Let x declared d body) Invalid
  | Becomes_simple s =>
    ∀ (x : variable) (declared : flambda_kind) (body : expr),
      expr_refines_under tenv (Let x declared d body)
        (Let x declared (Simple s) body)
  | Stays_with_type ty =>
    ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
      (vars : env) (v : value),
      env_satisfies_strictly tenv statics vars ->
      exec_named statics hp rd vars d = Some (v, hp', rd') ->
      type_satisfied statics vars v ty
  | Stays_with_kind kind =>
    ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
      (vars : env) (v : value),
      env_satisfies_strictly tenv statics vars ->
      exec_named statics hp rd vars d = Some (v, hp', rd') ->
      kind_of_value v = kind
  | Not_known => True
  end.

(* The two outcomes in which the defining expression stays are about its
   run on the heap. Where one that reads no cell
   (FlambdaHeapSpec.named_reads_heap) runs, its value is the one that
   [eval_named] gives it. *)
Lemma stays_with_type_pure (tenv : Typing_env.t) (d : named)
    (ty : flambda_type) :
  named_reads_heap d = false ->
  (∀ (statics : static_env) (vars : env) (v : value),
    env_satisfies_strictly tenv statics vars ->
    eval_named statics vars d = Some v ->
    type_satisfied statics vars v ty) ->
  outcome_sound tenv d (Stays_with_type ty).
Proof.
  intros Hpure Htyped statics hp hp' rd rd' vars v Hsat Hexec.
  exact (Htyped statics vars v Hsat
           (proj1 (exec_named_pure statics hp hp' rd rd' vars d v Hpure
                     Hexec))).
Qed.

Lemma stays_with_kind_pure (tenv : Typing_env.t) (d : named)
    (kind : flambda_kind) :
  named_reads_heap d = false ->
  (∀ (statics : static_env) (vars : env) (v : value),
    env_satisfies_strictly tenv statics vars ->
    eval_named statics vars d = Some v ->
    kind_of_value v = kind) ->
  outcome_sound tenv d (Stays_with_kind kind).
Proof.
  intros Hpure Hkind statics hp hp' rd rd' vars v Hsat Hexec.
  exact (Hkind statics vars v Hsat
           (proj1 (exec_named_pure statics hp hp' rd rd' vars d v Hpure
                     Hexec))).
Qed.

Lemma of_bool_shortcut_sound (tenv : Typing_env.t) (d : named)
    (answer : option (Meet_shortcut.t bool)) (otherwise : outcome) :
  (∀ (x : variable) (declared : flambda_kind) (body : expr),
    match answer with
    | Some (Meet_shortcut.Known_result b) =>
      expr_refines_under tenv (Let x declared d body)
        (Let x declared
           (Simple (Const (Naked_immediate (if b then 1 else 0)))) body)
    | Some Meet_shortcut.Invalid =>
      expr_equivalent_under tenv (Let x declared d body) Invalid
    | Some Meet_shortcut.Need_meet | None => True
    end) ->
  outcome_sound tenv d otherwise ->
  outcome_sound tenv d (of_bool_shortcut answer otherwise).
Proof.
  intros Hanswer Hotherwise.
  destruct answer as [[b | | ] | ]; simpl; try exact Hotherwise;
    exact Hanswer.
Qed.

Lemma of_simple_shortcut_sound (tenv : Typing_env.t) (d : named)
    (answer : option (Meet_shortcut.t simple)) (otherwise : outcome) :
  (∀ (x : variable) (declared : flambda_kind) (body : expr),
    match answer with
    | Some (Meet_shortcut.Known_result s) =>
      expr_refines_under tenv (Let x declared d body)
        (Let x declared (Simple s) body)
    | Some Meet_shortcut.Invalid =>
      expr_equivalent_under tenv (Let x declared d body) Invalid
    | Some Meet_shortcut.Need_meet | None => True
    end) ->
  outcome_sound tenv d otherwise ->
  outcome_sound tenv d (of_simple_shortcut answer otherwise).
Proof.
  intros Hanswer Hotherwise.
  destruct answer as [[s | | ] | ]; simpl; try exact Hotherwise;
    exact Hanswer.
Qed.

(* The value of a primitive is of its result kind, on whatever heap it
   is run. *)
Lemma prim_kind_sound (tenv : Typing_env.t) (p : prim) (args : list simple) :
  outcome_sound tenv (Prim p args) (Stays_with_kind (prim_result_kind p)).
Proof.
  intros statics hp hp' rd rd' vars v _ Hnamed.
  destruct (proj1 (exec_named_prim_some statics hp hp' rd rd' vars p args v)
              Hnamed) as (vs & _ & Hprim & _).
  exact (exec_prim_result_kind statics hp hp' p vs v Hprim).
Qed.

(* ======================================================================== *)
(* Unary primitives                                                         *)
(* ======================================================================== *)

Lemma simplify_unary_sound (tenv : Typing_env.t) (p : unary_primitive)
    (arg : simple) (ty : flambda_type) :
  argument_type tenv arg ty ->
  outcome_sound tenv (Prim (Unary p) [arg]) (simplify_unary tenv p arg ty).
Proof.
  intros Hty.
  assert (Hterms : simple_in_terms tenv arg).
  { intros x ->. exact (proj1 Hty). }
  pose proof (prim_kind_sound tenv (Unary p) [arg]) as Hunknown.
  (* By cases on the primitive, with no names given to what it holds:
     a primitive that has no line of its own in [simplify_unary]
     closes by [Hunknown]. *)
  destruct p; cbn [simplify_unary]; try exact Hunknown.
  - (* [Block_load] *)
    match goal with
    | |- context [Block_load ?kind ?mut ?field] =>
      destruct mut; try exact Hunknown;
        (apply of_simple_shortcut_sound; [ | exact Hunknown ]);
        intros x declared body;
        refine (let_block_load tenv x declared kind _ field arg ty
                  body _ Hty);
        discriminate
    end.
  - (* [Is_int] *)
    match goal with
    | |- context [Is_int ?variant_only] => destruct variant_only
    end.
    + apply of_bool_shortcut_sound; [ | exact Hunknown ].
      intros x declared body.
      exact (let_is_int_variant_only tenv x declared arg ty body Hty).
    + destruct (Provers.prove_is_int tenv ty) as [[b | ] | ] eqn:Hprove;
        try exact Hunknown.
      intros x declared body.
      exact (let_is_int_proved tenv x declared arg ty b body Hty
               Hprove).
  - (* [Is_null] *)
    apply of_bool_shortcut_sound; [ | exact Hunknown ].
    intros x declared body.
    exact (let_is_null tenv x declared arg ty body Hty).
  - (* [Get_tag] *)
    destruct (Provers.prove_is_int tenv ty) as [[[ | ] | ] | ] eqn:Hint.
    + intros x declared body.
      exact (let_get_tag_of_immediate tenv x declared arg ty body Hty
               Hint).
    + destruct (Provers.prove_get_tag tenv ty)
        as [[[ | tag [ | tag' tags]] | ] | ] eqn:Htag;
        try exact Hunknown.
      intros x declared body.
      refine (let_get_tag_proved tenv x declared arg ty [tag] tag
                body Hty Htag _).
      intros tag' [<- | []]. reflexivity.
    + destruct (Provers.prove_get_tag tenv ty)
        as [[[ | tag [ | tag' tags]] | ] | ] eqn:Htag;
        try exact Hunknown.
      intros x declared body.
      refine (let_get_tag_proved tenv x declared arg ty [tag] tag
                body Hty Htag _).
      intros tag' [<- | []]. reflexivity.
    + destruct (Provers.prove_get_tag tenv ty)
        as [[[ | tag [ | tag' tags]] | ] | ] eqn:Htag;
        try exact Hunknown.
      intros x declared body.
      refine (let_get_tag_proved tenv x declared arg ty [tag] tag
                body Hty Htag _).
      intros tag' [<- | []]. reflexivity.
  - (* [Unbox_number] *)
    apply of_simple_shortcut_sound; [ | exact Hunknown ].
    intros x declared body.
    exact (let_unbox_number tenv x declared _ arg body Hterms).
  - (* [Box_number] *)
    apply stays_with_type_pure; [ reflexivity | ].
    intros statics vars v Hsat Hnamed.
    exact (box_number_typed tenv statics vars _ arg ty v
             (env_satisfies_strictly_env_satisfies tenv statics vars Hsat)
             Hty Hnamed).
  - (* [Untag_immediate] *)
    destruct (types_well_formed_b tenv) eqn:Hwf; [ | exact Hunknown ].
    apply of_simple_shortcut_sound; [ | exact Hunknown ].
    intros x declared body.
    exact (let_untag_immediate tenv x declared arg body Hterms
             (proj1 (types_well_formed_b_iff tenv) Hwf)).
  - (* [Tag_immediate] *)
    apply stays_with_type_pure; [ reflexivity | ].
    intros statics vars v Hsat Hnamed.
    exact (tag_immediate_typed tenv statics vars arg ty v
             (env_satisfies_strictly_env_satisfies tenv statics vars Hsat)
             Hty Hnamed).
Qed.

(* ======================================================================== *)
(* The arithmetic of tagged immediates                                      *)
(* ======================================================================== *)

(* The tables for one side known: what they answer is right of the
   operation on a number in range. *)
Lemma op_rhs_unknown_sound (op : binary_int_arith_op) (lhs j r : Z) :
  in_range tagged_width j = true ->
  eval_int_arith tagged_width op lhs j = Some r ->
  match op_rhs_unknown op lhs with
  | Exactly result => r = result
  | The_other_side => r = j
  | No_result => False
  | Cannot_simplify => True
  end.
Proof.
  intros Hrange.
  destruct (int_arith_identities Standard_tagged_immediate j Hrange)
    as (Hadd0l & Hadd0r & Hsub0 & Hmul0l & Hmul0r & Hmul1l & Hmul1r & Handm1l
        & Handm1r & Hand0l & Hand0r & Horm1l & Horm1r & Hor0l & Hor0r
        & Hxor0l & Hxor0r & Hdivmod & Hmodm1).
  unfold op_rhs_unknown, symmetric_op_one_side_unknown, tagged_width.
  destruct op; try (intros _; exact I);
    repeat match goal with
           | |- context [?a =? ?b] => destruct (Z.eqb_spec a b) as [-> | _]
           end;
    try (intros _; exact I); congruence.
Qed.

Lemma op_lhs_unknown_sound (op : binary_int_arith_op) (rhs j r : Z) :
  in_range tagged_width j = true ->
  eval_int_arith tagged_width op j rhs = Some r ->
  match op_lhs_unknown op rhs with
  | Exactly result => r = result
  | The_other_side => r = j
  | No_result => False
  | Cannot_simplify => True
  end.
Proof.
  intros Hrange.
  destruct (int_arith_identities Standard_tagged_immediate j Hrange)
    as (Hadd0l & Hadd0r & Hsub0 & Hmul0l & Hmul0r & Hmul1l & Hmul1r & Handm1l
        & Handm1r & Hand0l & Hand0r & Horm1l & Horm1r & Hor0l & Hor0r
        & Hxor0l & Hxor0r & Hdivmod & Hmodm1).
  unfold op_lhs_unknown, symmetric_op_one_side_unknown, tagged_width.
  destruct op as [ | | | s | s | | | ];
    try (destruct (Hdivmod s) as (Hdiv0 & Hmod0 & Hdiv1 & Hmod1));
    try destruct s;
    repeat match goal with
           | |- context [?a =? ?b] => destruct (Z.eqb_spec a b) as [-> | _]
           end;
    try (intros _; exact I); congruence.
Qed.

Lemma one_side_results_in (table : Z -> one_side_outcome) (nums : list Z) :
  ∀ results : list possible_result,
    one_side_results table nums = Some results ->
    ∀ i : Z,
      i ∈ nums ->
      match table i with
      | Exactly result => Result_exactly result ∈ results
      | The_other_side => Result_other_side ∈ results
      | No_result => True
      | Cannot_simplify => False
      end.
Proof.
  induction nums as [ | n nums IH]; intros results; simpl.
  - intros _ i [].
  - destruct (one_side_results table nums) as [results' | ];
      [ | discriminate ].
    intros Hresults i [<- | Hin].
    + destruct (table n); try discriminate; try exact I;
        injection Hresults as <-; left; reflexivity.
    + specialize (IH results' eq_refl i Hin).
      destruct (table n); try discriminate; injection Hresults as <-;
        destruct (table i); try exact I; try contradiction;
        first [ exact IH | right; exact IH ].
Qed.

Lemma exact_results_in (results : list possible_result) (r : Z) :
  Result_exactly r ∈ results -> r ∈ exact_results results.
Proof.
  induction results as [ | [i | ] results IH]; simpl.
  - intros [].
  - intros [[= <-] | Hin]; [ left; reflexivity | right; exact (IH Hin) ].
  - intros [[=] | Hin]. exact (IH Hin).
Qed.

Section Int_arith.

Variable tenv : Typing_env.t.
Variable op : binary_int_arith_op.
Variables arg1 arg2 : simple.
Variables ty1 ty2 : flambda_type.
Hypothesis Hty1 : argument_type tenv arg1 ty1.
Hypothesis Hty2 : argument_type tenv arg2 ty2.

Let arith : named :=
  Prim (Binary (Int_arith Standard_tagged_immediate op)) [arg1; arg2].

Lemma int_arith_tagged_any :
  outcome_sound tenv arith (Stays_with_type any_tagged_immediate).
Proof.
  apply stays_with_type_pure; [ reflexivity | ].
  intros statics vars v Hsat Hnamed. unfold arith in Hnamed.
  simpl in Hnamed.
  destruct (eval_simple vars arg1) as [v1 | ]; [ | discriminate ].
  destruct (eval_simple vars arg2) as [v2 | ]; [ | discriminate ].
  change (eval_prim statics
            (Binary (Int_arith Standard_tagged_immediate op)) [v1; v2]
          = Some v) in Hnamed.
  destruct (int_arith_tagged_defined statics op v1 v2 v Hnamed)
    as [i1 [i2 [result [_ [_ [_ [_ [_ ->]]]]]]]].
  apply any_tagged_immediate_satisfied. exists result. reflexivity.
Qed.

Hypothesis Hwf1 : well_formed_input tenv ty1.
Hypothesis Hwf2 : well_formed_input tenv ty2.

Lemma check_possible_results_sound (first : bool)
    (results : list possible_result) :
  (∀ i1 i2 r : Z,
    possible_tagged_immediate tenv ty1 i1 ->
    possible_tagged_immediate tenv ty2 i2 ->
    eval_int_arith tagged_width op i1 i2 = Some r ->
    Result_exactly r ∈ results
    ∨ (Result_other_side ∈ results ∧ r = if first then i1 else i2)) ->
  outcome_sound tenv arith
    (check_possible_results results (if first then arg1 else arg2)).
Proof.
  intros Hresults.
  assert (Hvalue :
    ∀ (statics : static_env) (vars : env) (v : value),
      env_satisfies_strictly tenv statics vars ->
      eval_named statics vars arith = Some v ->
      ∃ i1 i2 r : Z,
        possible_tagged_immediate tenv ty1 i1
        ∧ possible_tagged_immediate tenv ty2 i2
        ∧ eval_simple vars arg1 = Some (Value_const (Tagged_immediate i1))
        ∧ eval_simple vars arg2 = Some (Value_const (Tagged_immediate i2))
        ∧ eval_int_arith tagged_width op i1 i2 = Some r
        ∧ v = Value_const (Tagged_immediate r)).
  { intros statics vars v Hsat Hnamed.
    exact (int_arith_tagged_possible tenv statics vars op arg1 arg2 ty1 ty2
             v Hsat Hty1 Hty2 Hwf1 Hwf2 Hnamed). }
  unfold check_possible_results.
  destruct results as [ | result rest].
  - intros x declared body. apply let_no_value_pure; [ reflexivity | ].
    intros statics vars v Hsat Hnamed.
    destruct (Hvalue statics vars v Hsat Hnamed)
      as (i1 & i2 & r & Hi1 & Hi2 & _ & _ & Heval & _).
    destruct (Hresults i1 i2 r Hi1 Hi2 Heval) as [[] | [[] _]].
  - destruct (forallb (possible_result_eqb result) rest) eqn:Hsame.
    + assert (Hall :
        ∀ r' : possible_result, r' ∈ result :: rest -> r' = result).
      { intros r' [<- | Hin]; [ reflexivity | ].
        rewrite forallb_forall in Hsame. specialize (Hsame r' Hin).
        destruct result, r'; simpl in Hsame; try discriminate;
          [ apply Z.eqb_eq in Hsame; subst; reflexivity | reflexivity ]. }
      destruct result as [i | ].
      * intros x declared body. apply let_constant_pure; [ reflexivity | ].
        intros statics vars v Hsat Hnamed.
        destruct (Hvalue statics vars v Hsat Hnamed)
          as (i1 & i2 & r & Hi1 & Hi2 & _ & _ & Heval & ->).
        destruct (Hresults i1 i2 r Hi1 Hi2 Heval) as [Hin | [Hin _]].
        -- injection (Hall _ Hin) as ->. reflexivity.
        -- discriminate (Hall _ Hin).
      * intros x declared body. apply let_simple_pure; [ reflexivity | ].
        intros statics vars v Hsat Hnamed.
        destruct (Hvalue statics vars v Hsat Hnamed)
          as (i1 & i2 & r & Hi1 & Hi2 & Hv1 & Hv2 & Heval & ->).
        destruct (Hresults i1 i2 r Hi1 Hi2 Heval) as [Hin | [_ ->]].
        -- discriminate (Hall _ Hin).
        -- destruct first; assumption.
    + destruct (forallb
                  (fun r =>
                     match r with
                     | Result_exactly _ => true
                     | Result_other_side => false
                     end)
                  (result :: rest)) eqn:Hexact.
      * apply stays_with_type_pure; [ reflexivity | ].
        intros statics vars v Hsat Hnamed.
        destruct (Hvalue statics vars v Hsat Hnamed)
          as (i1 & i2 & r & Hi1 & Hi2 & _ & _ & Heval & ->).
        apply these_tagged_immediates_satisfied. exists r.
        split; [ reflexivity | ].
        destruct (Hresults i1 i2 r Hi1 Hi2 Heval) as [Hin | [Hin _]].
        -- exact (exact_results_in _ _ Hin).
        -- rewrite forallb_forall in Hexact. discriminate (Hexact _ Hin).
      * exact int_arith_tagged_any.
Qed.

End Int_arith.

Lemma simplify_int_arith_tagged_sound (tenv : Typing_env.t)
    (op : binary_int_arith_op) (arg1 arg2 : simple) (ty1 ty2 : flambda_type) :
  argument_type tenv arg1 ty1 ->
  argument_type tenv arg2 ty2 ->
  outcome_sound tenv
    (Prim (Binary (Int_arith Standard_tagged_immediate op)) [arg1; arg2])
    (simplify_int_arith_tagged tenv op arg1 ty1 arg2 ty2).
Proof.
  intros Hty1 Hty2. unfold simplify_int_arith_tagged.
  pose proof (int_arith_tagged_any tenv op arg1 arg2) as Hunknown.
  destruct (well_formed_input_b tenv ty1 && well_formed_input_b tenv ty2)
    eqn:Hwf; [ | exact Hunknown ].
  apply andb_true_iff in Hwf. destruct Hwf as [Hwf1 Hwf2].
  apply well_formed_input_b_sound in Hwf1, Hwf2.
  pose proof (check_possible_results_sound tenv op arg1 arg2 ty1 ty2
                Hty1 Hty2 Hwf1 Hwf2) as Hcheck.
  (* Where a prover answers [Invalid] there is no number. *)
  assert (Hinvalid :
    (∀ i1 i2 : Z,
      possible_tagged_immediate tenv ty1 i1 ->
      possible_tagged_immediate tenv ty2 i2 ->
      False) ->
    outcome_sound tenv
      (Prim (Binary (Int_arith Standard_tagged_immediate op)) [arg1; arg2])
      Becomes_invalid).
  { intros Hnone x declared body. apply let_no_value_pure; [ reflexivity | ].
    intros statics vars v Hsat Hnamed.
    destruct (int_arith_tagged_possible tenv statics vars op arg1 arg2 ty1
                ty2 v Hsat Hty1 Hty2 Hwf1 Hwf2 Hnamed)
      as (i1 & i2 & r & Hi1 & Hi2 & _).
    exact (Hnone i1 i2 Hi1 Hi2). }
  unfold possible_tagged_immediate in Hcheck, Hinvalid.
  destruct (Provers.meet_equals_tagged_immediates tenv ty1)
    as [[nums1 | | ] | ],
    (Provers.meet_equals_tagged_immediates tenv ty2) as [[nums2 | | ] | ];
    try exact Hunknown;
    try (apply Hinvalid; intros i1 i2 [_ H1] [_ H2];
         first [ exact H1 | exact H2 ]).
  - (* Both are known. *)
    destruct ((max_num_possible_results <? length nums1)%nat
              || (max_num_possible_results <? length nums2)%nat);
      [ exact Hunknown | ].
    apply (Hcheck true). intros i1 i2 r [_ H1] [_ H2] Heval. left.
    unfold both_sides_results. apply in_flat_map. exists i1.
    split; [ exact H1 | ]. apply in_flat_map. exists i2.
    split; [ exact H2 | ]. rewrite Heval. left. reflexivity.
  - (* The first is known; the second needs a meet. *)
    destruct (one_side_results (op_rhs_unknown op) nums1) as [results | ]
      eqn:Hresults; [ | exact Hunknown ].
    apply (Hcheck false). intros i1 i2 r [_ H1] [Hrange2 _] Heval.
    pose proof (one_side_results_in _ _ _ Hresults i1 H1) as Hin.
    pose proof (op_rhs_unknown_sound op i1 i2 r Hrange2 Heval) as Htable.
    destruct (op_rhs_unknown op i1) as [result | | | ];
      [ left; subst; exact Hin | right; split; [ exact Hin | exact Htable ]
      | contradiction | contradiction ].
  - (* The first is known; the second stops. *)
    destruct (one_side_results (op_rhs_unknown op) nums1) as [results | ]
      eqn:Hresults; [ | exact Hunknown ].
    apply (Hcheck false). intros i1 i2 r [_ H1] [Hrange2 _] Heval.
    pose proof (one_side_results_in _ _ _ Hresults i1 H1) as Hin.
    pose proof (op_rhs_unknown_sound op i1 i2 r Hrange2 Heval) as Htable.
    destruct (op_rhs_unknown op i1) as [result | | | ];
      [ left; subst; exact Hin | right; split; [ exact Hin | exact Htable ]
      | contradiction | contradiction ].
  - (* The second is known; the first needs a meet. *)
    destruct (one_side_results (op_lhs_unknown op) nums2) as [results | ]
      eqn:Hresults; [ | exact Hunknown ].
    apply (Hcheck true). intros i1 i2 r [Hrange1 _] [_ H2] Heval.
    pose proof (one_side_results_in _ _ _ Hresults i2 H2) as Hin.
    pose proof (op_lhs_unknown_sound op i2 i1 r Hrange1 Heval) as Htable.
    destruct (op_lhs_unknown op i2) as [result | | | ];
      [ left; subst; exact Hin | right; split; [ exact Hin | exact Htable ]
      | contradiction | contradiction ].
  - (* The second is known; the first stops. *)
    destruct (one_side_results (op_lhs_unknown op) nums2) as [results | ]
      eqn:Hresults; [ | exact Hunknown ].
    apply (Hcheck true). intros i1 i2 r [Hrange1 _] [_ H2] Heval.
    pose proof (one_side_results_in _ _ _ Hresults i2 H2) as Hin.
    pose proof (op_lhs_unknown_sound op i2 i1 r Hrange1 Heval) as Htable.
    destruct (op_lhs_unknown op i2) as [result | | | ];
      [ left; subst; exact Hin | right; split; [ exact Hin | exact Htable ]
      | contradiction | contradiction ].
Qed.

(* A comparison that yields a boolean, and physical equality, have no
   value but the two booleans. *)
Lemma int_comp_yields_bool (statics : static_env) (kind : standard_int)
    (c : int_comparison) (vs : list value) (v : value) :
  eval_prim statics (Binary (Int_comp kind (Yielding_bool c))) vs = Some v ->
  v = Value_const (Naked_immediate 0) ∨ v = Value_const (Naked_immediate 1).
Proof.
  destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
  cbn [eval_prim eval_binary_primitive].
  destruct (int_of_value kind v1) as [a | ]; [ | discriminate ].
  destruct (int_of_value kind v2) as [b | ]; [ | discriminate ].
  cbn. intros [= <-].
  destruct (eval_int_comparison (int_width kind) c a b); auto.
Qed.

Lemma phys_equal_yields_bool (statics : static_env) (op : equality_comparison)
    (vs : list value) (v : value) :
  eval_prim statics (Binary (Phys_equal op)) vs = Some v ->
  v = Value_const (Naked_immediate 0) ∨ v = Value_const (Naked_immediate 1).
Proof.
  destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
  cbn [eval_prim eval_binary_primitive].
  destruct (eval_phys_equal statics v1 v2) as [same | ]; [ | discriminate ].
  cbn. intros [= <-]. destruct op, same; auto.
Qed.

Lemma yields_bool_sound (tenv : Typing_env.t) (d : named) :
  (∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
     (vars : env) (v : value),
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    v = Value_const (Naked_immediate 0)
    ∨ v = Value_const (Naked_immediate 1)) ->
  outcome_sound tenv d (Stays_with_type any_naked_bool).
Proof.
  intros Hbool statics hp hp' rd rd' vars v _ Hnamed.
  apply (proj2 (any_bool_satisfied statics vars v)).
  exact (Hbool statics hp hp' rd rd' vars v Hnamed).
Qed.

Lemma comparison_unknown_sound (tenv : Typing_env.t) (kind : standard_int)
    (behaviour : int_comparison_behaviour) (args : list simple) :
  outcome_sound tenv (Prim (Binary (Int_comp kind behaviour)) args)
    (comparison_unknown behaviour).
Proof.
  destruct behaviour as [c | s]; cbn [comparison_unknown].
  - apply yields_bool_sound. intros statics hp hp' rd rd' vars v Hexec.
    apply exec_named_pure in Hexec; [ | reflexivity ].
    destruct Hexec as [Hnamed _]. cbn [eval_named] in Hnamed.
    destruct (eval_simples vars args) as [vs | ]; [ | discriminate ].
    exact (int_comp_yields_bool statics kind c vs v Hnamed).
  - exact (prim_kind_sound tenv
             (Binary (Int_comp kind (Yielding_int_like_compare_functions s)))
             args).
Qed.

Lemma simplify_int_comp_tagged_sound (tenv : Typing_env.t)
    (behaviour : int_comparison_behaviour) (arg1 arg2 : simple)
    (ty1 ty2 : flambda_type) :
  argument_type tenv arg1 ty1 ->
  argument_type tenv arg2 ty2 ->
  outcome_sound tenv
    (Prim (Binary (Int_comp Standard_tagged_immediate behaviour))
       [arg1; arg2])
    (simplify_int_comp_tagged tenv behaviour ty1 ty2).
Proof.
  intros Hty1 Hty2. unfold simplify_int_comp_tagged.
  pose proof (comparison_unknown_sound tenv Standard_tagged_immediate
                behaviour [arg1; arg2]) as Hunknown.
  destruct (well_formed_input_b tenv ty1 && well_formed_input_b tenv ty2)
    eqn:Hwf; [ | exact Hunknown ].
  apply andb_true_iff in Hwf. destruct Hwf as [Hwf1 Hwf2].
  apply well_formed_input_b_sound in Hwf1, Hwf2.
  pose proof (fun x declared body =>
                let_int_comp_tagged tenv x declared behaviour arg1
                  arg2 ty1 ty2 body Hty1 Hty2 Hwf1 Hwf2) as Hlaw.
  cbv zeta in Hlaw. revert Hlaw.
  destruct (Provers.meet_equals_tagged_immediates tenv ty1)
    as [[nums1 | | ] | ],
    (Provers.meet_equals_tagged_immediates tenv ty2) as [[nums2 | | ] | ];
    intros Hlaw; try exact Hunknown; try exact Hlaw.
  match goal with
  | |- context [match ?results with _ => _ end] =>
    destruct results as [ | result rest] eqn:Hresults
  end; [ exact Hunknown | ].
  destruct (forallb (Z.eqb result) rest) eqn:Hsame; [ | exact Hunknown ].
  intros x declared body. apply (Hlaw x declared body result).
  intros i1 i2 Hi1 Hi2.
  assert (Hin :
    eval_int_comp tagged_width behaviour i1 i2 ∈ result :: rest).
  { rewrite <- Hresults. apply in_flat_map. exists i1.
    split; [ exact Hi1 | ]. apply in_map. exact Hi2. }
  destruct Hin as [Hfirst | Hin]; [ symmetry; exact Hfirst | ].
  rewrite forallb_forall in Hsame. symmetry. apply Z.eqb_eq.
  exact (Hsame _ Hin).
Qed.

Lemma simplify_phys_equal_sound (tenv : Typing_env.t)
    (op : equality_comparison) (arg1 arg2 : simple) (ty1 ty2 : flambda_type) :
  argument_type tenv arg1 ty1 ->
  argument_type tenv arg2 ty2 ->
  outcome_sound tenv (Prim (Binary (Phys_equal op)) [arg1; arg2])
    (simplify_phys_equal tenv op ty1 ty2).
Proof.
  intros Hty1 Hty2. unfold simplify_phys_equal.
  assert (Hunknown :
    outcome_sound tenv (Prim (Binary (Phys_equal op)) [arg1; arg2])
      (Stays_with_type any_naked_bool)).
  { apply yields_bool_sound. intros statics hp hp' rd rd' vars v Hexec.
    destruct (phys_equal_exec statics hp hp' rd rd' vars op arg1 arg2 v Hexec)
      as (v1 & v2 & _ & _ & _ & _ & [Hprim | (same & _ & ->)]).
    - exact (phys_equal_yields_bool statics op [v1; v2] v Hprim).
    - destruct op, same; auto. }
  destruct (well_formed_input_b tenv ty1 && well_formed_input_b tenv ty2)
    eqn:Hwf; [ | exact Hunknown ].
  apply andb_true_iff in Hwf. destruct Hwf as [Hwf1 Hwf2].
  apply well_formed_input_b_sound in Hwf1, Hwf2.
  destruct (Provers.prove_physical_equality tenv ty1 ty2) as [[b | ] | ]
    eqn:Hprove; try exact Hunknown.
  intros x declared body.
  exact (let_phys_equal tenv x declared op arg1 arg2 ty1 ty2 b body Hty1
           Hty2 Hwf1 Hwf2 Hprove).
Qed.

Lemma simplify_binary_sound (tenv : Typing_env.t) (p : binary_primitive)
    (arg1 arg2 : simple) (ty1 ty2 : flambda_type) :
  argument_type tenv arg1 ty1 ->
  argument_type tenv arg2 ty2 ->
  outcome_sound tenv (Prim (Binary p) [arg1; arg2])
    (simplify_binary tenv p arg1 ty1 arg2 ty2).
Proof.
  intros Hty1 Hty2.
  destruct p; try exact (prim_kind_sound tenv _ [arg1; arg2]);
    try exact (simplify_phys_equal_sound tenv _ arg1 arg2 ty1 ty2 Hty1 Hty2);
    match goal with
    | |- context [Int_arith ?kind _] => destruct kind
    | |- context [Int_comp ?kind _] => destruct kind
    end;
    try exact (prim_kind_sound tenv _ [arg1; arg2]);
    try apply comparison_unknown_sound.
  - exact (simplify_int_arith_tagged_sound tenv _ arg1 arg2 ty1 ty2
             Hty1 Hty2).
  - exact (simplify_int_comp_tagged_sound tenv _ arg1 arg2 ty1 ty2
             Hty1 Hty2).
Qed.

Lemma simplify_variadic_sound (tenv : Typing_env.t) (p : variadic_primitive)
    (args : list simple) (tys : list flambda_type) :
  Forall2 (argument_type tenv) args tys ->
  outcome_sound tenv (Prim (Variadic p) args) (simplify_variadic p tys).
Proof.
  intros Htys. destruct p as [[tag] | kind]; simpl.
  2: { exact (prim_kind_sound tenv (Variadic (Make_immutable_array kind))
                args). }
  destruct (tag <? no_scan_tag) eqn:Htag.
  - apply stays_with_type_pure; [ reflexivity | ].
    intros statics vars v Hsat Hnamed.
    exact (make_block_typed tenv statics vars tag args tys v
             (env_satisfies_strictly_env_satisfies tenv statics vars Hsat)
             Htys (proj1 (Z.ltb_lt _ _) Htag) Hnamed).
  - exact (prim_kind_sound tenv
             (Variadic (Make_block (Block_kind_values tag))) args).
Qed.

(* ======================================================================== *)
(* A defining expression                                                    *)
(* ======================================================================== *)

Lemma expr_refines_under_refl (tenv : Typing_env.t) (e : expr) :
  expr_refines_under tenv e e.
Proof. intros vars conts statics hp rd _. apply refines_refl. Qed.

(* The closure that a projection of a function slot answers is of the
   kind value. *)
Lemma project_function_slot_kind (statics : static_env)
    (move_from move_to : function_slot) (v r : value) :
  project_function_slot statics move_from move_to v = Some r ->
  kind_of_value r = Value.
Proof.
  unfold project_function_slot.
  repeat match goal with
         | |- context [match ?e with _ => _ end] =>
           destruct e; try discriminate
         end;
    intros [= <-]; reflexivity.
Qed.

Lemma simplify_named_sound (tenv : Typing_env.t) (d : named) :
  outcome_sound tenv d (simplify_named tenv d).
Proof.
  unfold simplify_named.
  destruct (named_changes_heap d) eqn:Hheap; [ exact I | ].
  destruct d
    as [s | p args | code value_slots | slot function_decls value_slots
        | slot closure | project_from slot closure
        | move_from move_to closure];
    cbn beta iota zeta.
  - intros x declared body. apply expr_refines_under_refl.
  - destruct (simplify_simples tenv args) as [answers | ] eqn:Hargs;
      [ | exact I ].
    destruct (simplify_simples_answers tenv args answers Hargs)
      as [Htys _].
    destruct (prim_args_well_kinded p
                (map FlambdaTypeGrammar.kind (map fst answers)))
      eqn:Hkinds.
    + (* By cases on the family of the primitive; a family that is not
         one of the three named here is left as it is. *)
      destruct p;
        [ destruct args as [ | arg [ | arg' args]]; try exact I;
          destruct (map fst answers) as [ | ty [ | ty' tys]]; try exact I;
          inversion Htys as [ | ? ? ? ? Hty _]; subst;
          exact (simplify_unary_sound tenv _ arg ty Hty)
        | destruct args as [ | arg1 [ | arg2 [ | arg3 args]]]; try exact I;
          destruct (map fst answers) as [ | ty1 [ | ty2 [ | ty3 tys]]];
          try exact I;
          inversion Htys as [ | ? ? ? ? Hty1 Hrest]; subst;
          inversion Hrest as [ | ? ? ? ? Hty2 _]; subst;
          exact (simplify_binary_sound tenv _ arg1 arg2 ty1 ty2 Hty1
                   Hty2)
        | match goal with
          | |- outcome_sound _ (Prim (Variadic ?q) _) ?answer =>
            replace answer with (simplify_variadic q (map fst answers))
              by (destruct args; destruct (map fst answers); reflexivity)
          end;
          exact (simplify_variadic_sound tenv _ args (map fst answers) Htys)
        | .. ];
        destruct args; destruct (map fst answers); exact I.
    + intros x declared body.
      refine (let_prim_kind_mismatch tenv x declared p args (map fst answers)
                body _ Htys Hkinds).
      intros q ->. discriminate Hheap.
  - destruct (simplify_simples tenv (map snd value_slots))
      as [answers | ] eqn:Hslots.
    + destruct (simplify_simples_answers tenv _ answers Hslots)
        as [Htys _].
      apply stays_with_type_pure; [ reflexivity | ].
      intros statics vars v Hsat Hnamed.
      exact (set_of_closures_typed tenv statics vars code value_slots
               (map fst answers) v
               (env_satisfies_strictly_env_satisfies tenv statics vars Hsat)
               Htys Hnamed).
    + apply stays_with_kind_pure; [ reflexivity | ].
      intros statics vars v _ Hnamed. simpl in Hnamed.
      destruct (eval_value_slots vars value_slots); [ | discriminate ].
      injection Hnamed as <-. reflexivity.
  - destruct (simplify_simples tenv (map snd value_slots))
      as [answers | ] eqn:Hslots.
    + destruct (simplify_simples_answers tenv _ answers Hslots)
        as [Htys _].
      apply stays_with_type_pure; [ reflexivity | ].
      intros statics vars v Hsat Hnamed.
      exact (closure_of_set_typed tenv statics vars slot function_decls
               value_slots (map fst answers) v
               (env_satisfies_strictly_env_satisfies tenv statics vars Hsat)
               Htys Hnamed).
    + apply stays_with_kind_pure; [ reflexivity | ].
      intros statics vars v _ Hnamed. simpl in Hnamed.
      destruct (lookup slot function_decls); [ | discriminate ].
      simpl in Hnamed.
      destruct (eval_value_slots vars value_slots); [ | discriminate ].
      injection Hnamed as <-. reflexivity.
  - exact I.
  - exact I.
  - destruct (simplify_simple tenv closure) as [[ty canonical] | ]
      eqn:Hclosure; [ | exact I ].
    destruct (simplify_simple_answer tenv closure canonical ty Hclosure)
      as [Hty _].
    apply of_simple_shortcut_sound.
    + intros x declared body.
      exact (let_project_function_slot tenv x declared move_from
               move_to closure ty body Hty).
    + apply stays_with_kind_pure; [ reflexivity | ].
      intros statics vars v _ Hnamed. simpl in Hnamed.
      destruct (eval_simple vars closure) as [w | ]; [ | discriminate ].
      exact (project_function_slot_kind statics move_from move_to w v
               Hnamed).
Qed.

(* The arguments of a primitive may be replaced by their canonical
   elements. *)
Lemma canonical_named_equivalent (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (d : named) (body : expr) :
  expr_equivalent_under tenv (Let x declared d body)
    (Let x declared (canonical_named tenv d) body).
Proof.
  assert (Hrefl :
    expr_equivalent_under tenv (Let x declared d body)
      (Let x declared d body)).
  { intros vars conts statics hp rd _. apply equivalent_refl. }
  destruct d as [s | p args | | | | | ]; try exact Hrefl.
  cbn [canonical_named].
  destruct (simplify_simples tenv args) as [answers | ] eqn:Hargs;
    [ | exact Hrefl ].
  apply let_prim_canonical.
  exact (proj2 (simplify_simples_answers tenv args answers Hargs)).
Qed.

(* ======================================================================== *)
(* A [Let]                                                                  *)
(* ======================================================================== *)

(* [simple_in_terms_b_sound] and [simple_of_kind_b_sound] are in
   FlambdaSimplifyUsesProofs.v. *)

Lemma bind_simple_sound (tenv tenv' : Typing_env.t) (x : variable)
    (s : simple) (d' : named) :
  typing_env_well_formed tenv = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  bind_simple tenv x s = Step_let d' tenv' ->
  ∀ (declared : flambda_kind) (body body' : expr),
    expr_refines_under tenv' body body' ->
    expr_refines_under tenv (Let x declared (Simple s) body)
      (Let x declared d' body').
Proof.
  intros Hwf Hfresh. unfold bind_simple.
  destruct (simplify_simple tenv s) as [[ty canonical] | ] eqn:Hs;
    [ | discriminate ].
  destruct (simplify_simple_answer tenv s canonical ty Hs)
    as [_ Hcanonical].
  destruct (simple_in_terms_b tenv canonical
            && simple_of_kind_b tenv canonical (FlambdaTypeGrammar.kind ty))
    eqn:Htests; [ | discriminate ].
  apply andb_true_iff in Htests. destruct Htests as [Hterms Hkind].
  destruct (Typing_env.add_alias _ _ _) as [[ | result | ] | ] eqn:Hadd;
    try discriminate.
  intros [= <- <-] declared body body' Hbody.
  eapply expr_refines_under_trans.
  - apply expr_equivalent_under_refines.
    exact (let_simple_canonical tenv x declared s canonical body Hcanonical).
  - exact (let_simple_refines_under tenv x declared canonical body body'
             (FlambdaTypeGrammar.kind ty) result Hwf Hfresh
             (simple_in_terms_b_sound tenv canonical Hterms)
             (simple_of_kind_b_sound tenv canonical _ Hkind) Hadd Hbody).
Qed.

Lemma simplify_let_sound (tenv : Typing_env.t) (x : variable) (d : named) :
  match simplify_let tenv x d with
  | Step_invalid =>
    ∀ (declared : flambda_kind) (body : expr),
      expr_equivalent_under tenv (Let x declared d body) Invalid
  | Step_let d' tenv' =>
    ∀ (declared : flambda_kind) (body body' : expr),
      expr_refines_under tenv' body body' ->
      expr_refines_under tenv (Let x declared d body)
        (Let x declared d' body')
  | Step_stop => True
  end.
Proof.
  unfold simplify_let.
  destruct (typing_env_well_formed tenv
            && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
    [ | exact I ].
  apply andb_true_iff in Htests. destruct Htests as [Hwf Hfresh].
  apply negb_true_iff in Hfresh.
  pose proof (simplify_named_sound tenv d) as Hsound.
  destruct (simplify_named tenv d) as [ | s | ty | kind | ]; simpl in Hsound.
  - intros declared body. exact (Hsound x declared body).
  - destruct (bind_simple tenv x s) as [ | d' tenv' | ] eqn:Hbind;
      try exact I.
    + (* [bind_simple] does not answer [Step_invalid]. *)
      exfalso. revert Hbind. unfold bind_simple.
      repeat match goal with
             | |- context [match ?e with _ => _ end] =>
               destruct e; try discriminate
             end.
    + intros declared body body' Hbody.
      eapply expr_refines_under_trans; [ exact (Hsound x declared body) | ].
      exact (bind_simple_sound tenv tenv' x s d' Hwf Hfresh Hbind declared
               body body' Hbody).
  - destruct (type_in_scope_b tenv ty) eqn:Hscope; [ | exact I ].
    intros declared body body' Hbody.
    eapply expr_refines_under_trans;
      [ | apply expr_equivalent_under_refines;
          exact (canonical_named_equivalent tenv x declared d body') ].
    apply (let_typed_refines_under tenv x declared d body body' ty Hwf
             Hfresh);
      [ | exact Hsound | exact Hbody ].
    intros y Hy. unfold type_in_scope_b in Hscope.
    rewrite forallb_forall in Hscope. exact (Hscope y Hy).
  - intros declared body body' Hbody.
    eapply expr_refines_under_trans;
      [ | apply expr_equivalent_under_refines;
          exact (canonical_named_equivalent tenv x declared d body') ].
    exact (let_refines_under tenv x declared d body body' kind Hwf Hfresh
             Hsound Hbody).
  - exact I.
Qed.

(* ======================================================================== *)
(* The other expressions                                                    *)
(* ======================================================================== *)

Lemma simplify_apply_cont_equivalent (tenv : Typing_env.t) (ac : apply_cont) :
  expr_equivalent_under tenv (Apply_cont ac)
    (Apply_cont (simplify_apply_cont tenv ac)).
Proof.
  unfold simplify_apply_cont.
  destruct (simplify_simples tenv (ac_args ac)) as [answers | ] eqn:Hargs.
  - destruct ac as [k args action]. simpl in *.
    apply apply_cont_canonical.
    exact (proj2 (simplify_simples_answers tenv args answers Hargs)).
  - intros vars conts statics hp rd _. apply equivalent_refl.
Qed.

Lemma simplify_switch_refines (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  expr_refines_under tenv (Switch scrutinee arms)
    (simplify_switch tenv scrutinee arms).
Proof.
  unfold simplify_switch.
  destruct (simplify_simple tenv scrutinee) as [[ty canonical] | ]
    eqn:Hscrutinee; [ | apply expr_refines_under_refl ].
  destruct canonical as [x | sym | c]; try apply expr_refines_under_refl.
  destruct c; try apply expr_refines_under_refl.
  destruct (find_arm _ arms) as [ac | ] eqn:Harm;
    [ | apply expr_refines_under_refl ].
  destruct (simplify_simple_answer tenv scrutinee _ ty Hscrutinee)
    as [_ Hcanonical].
  eapply expr_refines_under_trans; apply expr_equivalent_under_refines.
  - exact (switch_known tenv scrutinee _ arms ac Hcanonical Harm).
  - apply simplify_apply_cont_equivalent.
Qed.

(* Under a [Let_static] the body runs with the same variables, in a
   static environment that has grown, and the symbols that it binds
   are defined. *)
Lemma add_symbol_definitions_env_satisfies_strictly (tenv : Typing_env.t)
    (syms : list symbol) (statics : static_env) (vars : env) :
  env_satisfies_strictly tenv statics vars ->
  env_satisfies_strictly (Typing_env.add_symbol_definitions tenv syms)
    statics vars.
Proof.
  intros [Hbound [valuation [Hsatisfied Hagrees]]].
  split; [ exact Hbound | ]. exists valuation.
  split; [ | exact Hagrees ].
  clear Hbound Hagrees. induction syms as [ | sym syms IH].
  - destruct tenv. exact Hsatisfied.
  - exact (proj2 (add_symbol_definition_satisfied _ statics valuation
                    (Typing_env.add_symbol_definitions tenv syms) sym) IH).
Qed.

(* A group of static bindings whose code is rewritten gives the
   symbols the same types: the environment reads the symbols only. *)
Lemma add_static_bindings_same_symbols
    (g : static_binding -> static_binding) :
  (∀ (s : symbol) (const : static_const),
    g (Bind_symbol s const) = Bind_symbol s const) ->
  (∀ (id : code_id) (c : code),
    ∃ c' : code, g (Bind_code id c) = Bind_code id c') ->
  (∀ id : code_id, g (Bind_deleted_code id) = Bind_deleted_code id) ->
  ∀ (tenv : Typing_env.t) (bindings : list static_binding),
    add_static_bindings tenv (map g bindings)
    = add_static_bindings tenv bindings.
Proof.
  intros Hsymbol Hcode Hdeleted tenv bindings. unfold add_static_bindings.
  assert (Hclosures :
    closure_symbols_of (map g bindings) = closure_symbols_of bindings).
  { induction bindings as [ | [id c | id | s const] bindings IH];
      cbn [map]; [ reflexivity | | | ].
    - destruct (Hcode id c) as [c' ->]. exact IH.
    - rewrite Hdeleted. exact IH.
    - rewrite Hsymbol. cbn [closure_symbols_of]. rewrite IH. reflexivity. }
  assert (Hblocks :
    ∀ tenv : Typing_env.t,
      add_block_like_bindings tenv (map g bindings)
      = add_block_like_bindings tenv bindings).
  { clear Hclosures tenv.
    induction bindings as [ | [id c | id | s const] bindings IH];
      intros tenv; cbn [map]; [ reflexivity | | | ].
    - destruct (Hcode id c) as [c' ->]. exact (IH tenv).
    - rewrite Hdeleted. exact (IH tenv).
    - rewrite Hsymbol. cbn [add_block_like_bindings]. apply IH. }
  assert (Hlast :
    ∀ prior tenv : Typing_env.t,
      add_closure_bindings prior tenv (map g bindings)
      = add_closure_bindings prior tenv bindings).
  { clear Hclosures Hblocks tenv.
    induction bindings as [ | [id c | id | s const] bindings IH];
      intros prior tenv; cbn [map]; [ reflexivity | | | ].
    - destruct (Hcode id c) as [c' ->]. exact (IH prior tenv).
    - rewrite Hdeleted. exact (IH prior tenv).
    - rewrite Hsymbol. cbn [add_closure_bindings]. apply IH. }
  cbv zeta. rewrite Hclosures, Hblocks. apply Hlast.
Qed.

Lemma add_static_bindings_simplify (tenv tenv' : Typing_env.t)
    (bindings : list static_binding) :
  add_static_bindings tenv (map (simplify_static_binding tenv') bindings)
  = add_static_bindings tenv bindings.
Proof.
  apply add_static_bindings_same_symbols.
  - intros s const. reflexivity.
  - intros id c. eexists. reflexivity.
  - intros id. reflexivity.
Qed.

(* ======================================================================== *)
(* A term                                                                   *)
(* ======================================================================== *)

(* The projection of a value slot replaced by the [Simple] that the
   prover answers. *)
Lemma resolve_value_slot_refines (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (d : named) (body : expr) :
  expr_refines_under tenv (Let x declared d body)
    (Let x declared (resolve_value_slot tenv declared d) body).
Proof.
  destruct d
    as [s | p args | code value_slots | slot function_decls value_slots
        | slot closure | project_from slot closure
        | move_from move_to closure];
    cbn [resolve_value_slot]; try apply expr_refines_under_refl;
    (destruct (simplify_simple tenv closure) as [[ty canonical] | ]
       eqn:Hclosure; [ | apply expr_refines_under_refl ]);
    (destruct (simplify_simple_answer tenv closure canonical ty Hclosure)
       as [Hty _]);
    (destruct (simplify_project_value_slot tenv declared slot ty) as [s | ]
       eqn:Hslot; [ | apply expr_refines_under_refl ]).
  - exact (let_project_value_slot_known tenv x declared slot declared
             closure ty s body Hty Hslot).
  - exact (let_project_value_slot_from_known tenv x declared project_from
             slot declared closure ty s body Hty Hslot).
Qed.

(* ======================================================================== *)
(* Where a continuation is applied                                          *)
(* ======================================================================== *)

(* A [Let] that refines the same [Let] of another defining expression,
   whatever the body: the other defining expression does on the heap
   what the first does, where the value is of the kind of the [Let]: it
   has the same value and leaves the same heap. (The record that it
   leaves is not seen: an exit does not show it.) The body that shows
   it gives the variable to a way out of the term. *)
Lemma let_refines_value (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (d d' : named) :
  (∀ body : expr,
    expr_refines_under tenv (Let x declared d body) (Let x declared d' body))
  ->
  ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
    (vars : env) (v : value),
    env_satisfies_strictly tenv statics vars ->
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    kind_of_value v = declared ->
    ∃ rd'' : read_record,
      exec_named statics hp rd vars d' = Some (v, hp', rd'').
Proof.
  intros Hrefines statics hp hp' rd rd' vars v Hsat Hd Hkind.
  set (probe := Apply_cont (Apply_cont_expr 1%positive [Var x] None)).
  assert (Hprobe :
    ∀ (w : value) (h : heap) (r : read_record),
      behaves (State probe ((x, w) :: vars) [] statics h r)
        (Exits 1%positive [w] h)).
  { intros w h r. apply step_behaves. unfold probe. rewrite step_apply_cont.
    cbn [ac_args eval_simples eval_simple]. rewrite lookup_head.
    reflexivity. }
  destruct (Hrefines probe vars [] statics hp rd Hsat) as [Hexits _].
  assert (Hsource :
    behaves (State (Let x declared d probe) vars [] statics hp rd)
      (Exits 1%positive [v] hp')).
  { apply step_behaves. rewrite step_let_exec, Hd.
    rewrite (proj2 (kind_eqb_eq _ _) Hkind). exact (Hprobe v hp' rd'). }
  apply Hexits in Hsource. apply step_behaves in Hsource.
  rewrite step_let_exec in Hsource.
  destruct (exec_named statics hp rd vars d') as [[[v' hp''] rd''] | ];
    [ | discriminate Hsource ].
  destruct (kind_eqb (kind_of_value v') declared); [ | discriminate Hsource ].
  simpl in Hsource.
  pose proof
    (behaviour_unique (State probe ((x, v') :: vars) [] statics hp'' rd'')
       (Exits 1%positive [v] hp') (Exits 1%positive [v'] hp'')
       Hsource (Hprobe v' hp'' rd''))
    as Heq.
  injection Heq as <- <-. exists rd''. reflexivity.
Qed.

Lemma bind_simple_satisfied (tenv tenv' : Typing_env.t) (x : variable)
    (s : simple) (d' : named) (statics : static_env) (vars : env)
    (v : value) :
  typing_env_well_formed tenv = true ->
  Typing_env.mem None tenv (Name_var x) = false ->
  bind_simple tenv x s = Step_let d' tenv' ->
  env_satisfies_strictly tenv statics vars ->
  eval_simple vars s = Some v ->
  env_satisfies_strictly tenv' statics ((x, v) :: vars).
Proof.
  intros Hwf Hfresh. unfold bind_simple.
  destruct (simplify_simple tenv s) as [[ty canonical] | ] eqn:Hs;
    [ | discriminate ].
  destruct (simplify_simple_answer tenv s canonical ty Hs)
    as [_ Hcanonical].
  destruct (simple_in_terms_b tenv canonical
            && simple_of_kind_b tenv canonical (FlambdaTypeGrammar.kind ty))
    eqn:Htests; [ | discriminate ].
  apply andb_true_iff in Htests. destruct Htests as [Hterms Hkind].
  destruct (Typing_env.add_alias _ _ _) as [[ | result | ] | ] eqn:Hadd;
    try discriminate.
  intros [= <- <-] Hsat Hv.
  pose proof (simple_in_terms_b_sound tenv canonical Hterms) as Hscope.
  assert (Hcanonical_value : eval_simple vars canonical = Some v).
  { rewrite (canonical_in_terms_sound tenv statics vars s canonical Hsat
               Hcanonical).
    exact Hv. }
  assert (Hother :
    eval_simple ((x, v) :: vars) canonical = eval_simple vars canonical).
  { destruct canonical as [y | sym | c]; [ | reflexivity.. ]. simpl.
    apply lookup_other.
    exact (not_bound_not_in_scope tenv x y Hfresh (Hscope y eq_refl)). }
  apply (add_alias_env_satisfies_strictly
           (Typing_env.add_variable_definition tenv x
              (FlambdaTypeGrammar.kind ty) Name_mode.normal)
           statics ((x, v) :: vars) (Var x) canonical result).
  - intros y [[= <-] | ->];
      apply (add_variable_definition_in_scope tenv x _ _ Hwf);
      [ left; reflexivity | right ].
    split; [ | exact (Hscope y eq_refl) ].
    exact (not_bound_not_in_scope tenv x y Hfresh (Hscope y eq_refl)).
  - exact Hadd.
  - rewrite Hother, Hcanonical_value. simpl. apply lookup_head.
  - apply add_variable_definition_env_satisfies_strictly;
      [ exact Hwf | exact Hfresh | | exact Hsat ].
    apply (simple_of_kind_value tenv statics vars canonical _ v);
      [ | exact Hscope | exact (simple_of_kind_b_sound tenv canonical _ Hkind)
        | exact Hcanonical_value ].
    apply env_satisfies_strictly_env_satisfies. exact Hsat.
Qed.

Lemma simplify_let_satisfied (tenv tenv' : Typing_env.t) (x : variable)
    (d d' : named) (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (v : value) :
  simplify_let tenv x d = Step_let d' tenv' ->
  env_satisfies_strictly tenv statics vars ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  env_satisfies_strictly tenv' statics ((x, v) :: vars).
Proof.
  unfold simplify_let.
  destruct (typing_env_well_formed tenv
            && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
    [ | discriminate ].
  apply andb_true_iff in Htests. destruct Htests as [Hwf Hfresh].
  apply negb_true_iff in Hfresh.
  pose proof (simplify_named_sound tenv d) as Hsound.
  destruct (simplify_named tenv d) as [ | s | ty | kind | ]; simpl in Hsound;
    try discriminate.
  - intros Hbind Hsat Hv.
    apply (bind_simple_satisfied tenv tenv' x s d' statics vars v Hwf Hfresh
             Hbind Hsat).
    destruct (let_refines_value tenv x (kind_of_value v) d (Simple s)
                (fun body => Hsound x (kind_of_value v) body) statics hp hp'
                rd rd' vars v Hsat Hv eq_refl) as [rd'' Hs].
    simpl in Hs.
    destruct (eval_simple vars s) as [w | ]; [ | discriminate ].
    injection Hs as -> _ _. reflexivity.
  - destruct (type_in_scope_b tenv ty) eqn:Hscope; [ | discriminate ].
    intros [= _ <-] Hsat Hv.
    pose proof (Hsound statics hp hp' rd rd' vars v Hsat Hv) as Hty.
    assert (Hfree :
      ∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y).
    { intros y Hy. unfold type_in_scope_b in Hscope.
      rewrite forallb_forall in Hscope. exact (Hscope y Hy). }
    apply replace_equation_env_satisfies_strictly.
    + apply (add_variable_definition_in_scope tenv x _ x Hwf). left.
      reflexivity.
    + intros y Hin. apply (add_variable_definition_in_scope tenv x _ y Hwf).
      right. split; [ | exact (Hfree y Hin) ].
      exact (not_bound_not_in_scope tenv x y Hfresh (Hfree y Hin)).
    + intros v' Hlookup.
      pose proof (eq_trans (eq_sym Hlookup) (lookup_head x v vars)) as Heq.
      injection Heq as ->.
      apply (type_satisfied_coincide statics vars ((x, v) :: vars) v ty);
        [ | exact Hty ].
      intros y Hin. symmetry. apply lookup_other.
      exact (not_bound_not_in_scope tenv x y Hfresh (Hfree y Hin)).
    + apply add_variable_definition_env_satisfies_strictly;
        [ exact Hwf | exact Hfresh | | exact Hsat ].
      exact (type_satisfied_kind statics vars v ty Hty).
  - intros [= _ <-] Hsat Hv.
    apply add_variable_definition_env_satisfies_strictly;
      [ exact Hwf | exact Hfresh
      | exact (Hsound statics hp hp' rd rd' vars v Hsat Hv) | exact Hsat ].
Qed.

Theorem let_env_satisfied : let_env_satisfied_statement.
Proof.
  intros tenv tenv' x declared d statics hp hp' rd rd' vars v Henv Hsat Hv
    Hkind.
  unfold let_env in Henv.
  destruct (let_refines_value tenv x declared d _
              (resolve_value_slot_refines tenv x declared d) statics hp hp'
              rd rd' vars v Hsat Hv Hkind)
    as [rd'' Hresolved].
  destruct (simplify_let tenv x (resolve_value_slot tenv declared d))
    as [ | d' tenv'' | ] eqn:Hlet; [ discriminate | | ].
  - injection Henv as <-.
    exact (simplify_let_satisfied tenv tenv'' x _ d' statics hp hp' rd rd''
             vars v Hlet Hsat Hresolved).
  - unfold declare_variable in Henv.
    destruct (typing_env_well_formed tenv
              && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
      [ | discriminate ].
    injection Henv as <-. apply andb_true_iff in Htests.
    destruct Htests as [Hwf Hfresh]. apply negb_true_iff in Hfresh.
    apply add_variable_definition_env_satisfies_strictly;
      [ exact Hwf | exact Hfresh | exact Hkind | exact Hsat ].
Qed.

(* By recursion on the body: the [Let_cont] is moved down past each
   binder, to the application (FlambdaSinkContsProofs.v), the handler
   is rewritten there, and the [Let_cont] is moved back. *)
Fixpoint find_use_refines_aux (body : expr) {struct body} :
  ∀ (scope tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' : expr)
    (target : Typing_env.t) (xs : list variable) (ks : list continuation),
    find_use scope tenv k (Handler params metadata handler) body
    = Some (target, xs, ks) ->
    binders_not_free xs ks (Handler params metadata handler') = true ->
    expr_refines_under target handler handler' ->
    expr_refines_under tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  destruct body
    as [x declared d rest | bindings rest | k' h' rest | handlers rest | ap
        | ac | scrutinee arms | ];
    intros scope tenv k params metadata handler handler' target xs ks Hfind
      Hfree Hhandler;
    cbn [find_use] in Hfind; try discriminate.
  - destruct (var_free_in_handler_b x (Handler params metadata handler))
      eqn:Hx; [ discriminate | ].
    destruct (let_env tenv x declared d) as [tenv' | ] eqn:Henv;
      [ | discriminate ].
    destruct (find_use scope tenv' k (Handler params metadata handler) rest)
      as [[[target' xs'] ks'] | ] eqn:Hrest; [ | discriminate ].
    injection Hfind as <- <- <-.
    unfold binders_not_free in Hfree. cbn [forallb] in Hfree.
    apply andb_true_iff in Hfree. destruct Hfree as [Hxs Hks].
    apply andb_true_iff in Hxs. destruct Hxs as [Hx' Hxs].
    apply negb_true_iff in Hx'.
    assert (Hrest_free :
      binders_not_free xs' ks' (Handler params metadata handler') = true).
    { unfold binders_not_free. rewrite Hxs, Hks. reflexivity. }
    pose proof (find_use_refines_aux rest scope tenv' k params metadata
                  handler handler' target' xs' ks' Hrest Hrest_free Hhandler)
      as IH.
    intros vars conts statics hp rd Hsat.
    apply (refines_trans _
             (State
                (Let x declared d
                   (Let_cont_non_recursive k
                      (Handler params metadata handler) rest))
                vars conts statics hp rd)).
    { apply equivalent_refines, let_cont_past_let. intros Hfree'.
      apply var_free_in_handler_b_sound in Hfree'. congruence. }
    apply (refines_trans _
             (State
                (Let x declared d
                   (Let_cont_non_recursive k
                      (Handler params metadata handler') rest))
                vars conts statics hp rd)).
    { apply let_refines_step_exec; [ reflexivity | intros v hp' rd' Hv Hkind ].
      apply IH.
      exact (let_env_satisfied tenv tenv' x declared d statics hp hp' rd rd'
               vars v Henv Hsat Hv Hkind). }
    apply equivalent_refines, equivalent_sym, let_cont_past_let.
    intros Hfree'. apply var_free_in_handler_b_sound in Hfree'. congruence.
  - destruct (negb (Pos.eqb k k') && negb (cont_free_in_handler_b k h')
              && negb
                   (cont_free_in_handler_b k'
                      (Handler params metadata handler)))
      eqn:Htests; [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [Htests Hk'].
    apply andb_true_iff in Htests. destruct Htests as [Hne Hk].
    apply negb_true_iff in Hne, Hk, Hk'. apply Pos.eqb_neq in Hne.
    destruct (find_use scope tenv k (Handler params metadata handler) rest)
      as [[[target' xs'] ks'] | ] eqn:Hrest; [ | discriminate ].
    injection Hfind as <- <- <-.
    unfold binders_not_free in Hfree. cbn [forallb] in Hfree.
    apply andb_true_iff in Hfree. destruct Hfree as [Hxs Hks].
    apply andb_true_iff in Hks. destruct Hks as [Hk'' Hks].
    apply negb_true_iff in Hk''.
    assert (Hrest_free :
      binders_not_free xs' ks' (Handler params metadata handler') = true).
    { unfold binders_not_free. rewrite Hxs, Hks. reflexivity. }
    pose proof (find_use_refines_aux rest scope tenv k params metadata
                  handler handler' target' xs' ks' Hrest Hrest_free Hhandler)
      as IH.
    intros vars conts statics hp rd Hsat.
    apply (refines_trans _
             (State
                (Let_cont_non_recursive k' h'
                   (Let_cont_non_recursive k
                      (Handler params metadata handler) rest))
                vars conts statics hp rd)).
    { apply equivalent_refines, let_cont_past_let_cont; [ exact Hne | | ];
        intros Hfree'; apply cont_free_in_handler_b_sound in Hfree';
        congruence. }
    apply (refines_trans _
             (State
                (Let_cont_non_recursive k' h'
                   (Let_cont_non_recursive k
                      (Handler params metadata handler') rest))
                vars conts statics hp rd)).
    { exact (let_cont_body_refines_under tenv k' h' _ _ IH vars conts statics
               hp rd Hsat). }
    apply equivalent_refines, equivalent_sym, let_cont_past_let_cont;
      [ exact Hne | | ];
      intros Hfree'; apply cont_free_in_handler_b_sound in Hfree';
      congruence.
  - destruct (leaf_uses tenv k (Apply_cont ac))
      as [[ | [tenv_use args] [ | second others]] | ] eqn:Hleaf;
      try discriminate.
    destruct (use_env scope tenv_use params metadata args) as [target' | ]
      eqn:Huse; [ | discriminate ].
    injection Hfind as <- _ _.
    exact (let_cont_handler_at_use scope tenv k params metadata handler
             handler' _ tenv_use args target' Hleaf Huse Hhandler).
  - destruct (leaf_uses tenv k (Switch scrutinee arms))
      as [[ | [tenv_use args] [ | second others]] | ] eqn:Hleaf;
      try discriminate.
    destruct (use_env scope tenv_use params metadata args) as [target' | ]
      eqn:Huse; [ | discriminate ].
    injection Hfind as <- _ _.
    exact (let_cont_handler_at_use scope tenv k params metadata handler
             handler' _ tenv_use args target' Hleaf Huse Hhandler).
Qed.

Theorem find_use_refines : find_use_refines_statement.
Proof.
  intros scope tenv k params metadata handler handler' body.
  exact (find_use_refines_aux body scope tenv k params metadata handler
           handler').
Qed.

(* ======================================================================== *)
(* Several applications                                                     *)
(* ======================================================================== *)

(* A [Let_cont] copied to the head of the handler of the next one, the
   original being kept: the first step of
   FlambdaSinkContsProofs.let_cont_into_handler, which then drops the
   original. Here the original stays, for the applications that are
   not in that handler. *)
Lemma let_cont_copied_into_handler (k k' : continuation)
    (h : continuation_handler) (params' : list variable)
    (metadata' : handler_metadata) (handler' body : expr) :
  ~ cont_free_in_handler k h ->
  (∀ param : variable, param ∈ params' -> ~ var_free_in_handler param h) ->
  expr_equivalent
    (Let_cont_non_recursive k h
       (Let_cont_non_recursive k' (Handler params' metadata' handler') body))
    (Let_cont_non_recursive k h
       (Let_cont_non_recursive k'
          (Handler params' metadata' (Let_cont_non_recursive k h handler'))
          body)).
Proof.
  intros Hh Hparams vars conts statics hp rd.
  apply let_cont_non_recursive_equivalent_step.
  apply let_cont_handler_alike. intros vs statics' hp' rd'. simpl.
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata'));
    simpl; [ | exact I ].
  destruct (bind_params params' vs vars) as [handler_vars | ] eqn:Hbind;
    simpl; [ | exact I ].
  apply equivalent_sym.
  eapply equivalent_trans;
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  apply coincide_equivalent; [ reflexivity | ].
  intros k'' _. unfold conts_agree_at.
  change (rename_continuation identity_renaming k'') with k''.
  destruct (Pos.eq_dec k'' k) as [-> | Hne];
    [ | rewrite (resolve_other k'' k) by exact Hne; apply Same_matches ].
  rewrite !resolve_head. apply closures_match.
  - intros x Hx.
    apply (lookup_bind_params_other params' vs vars handler_vars x Hbind).
    intros Hin. exact (Hparams x Hin Hx).
  - intros k1 Hk1. rewrite resolve_other; [ apply Same_matches | ].
    intros ->. contradiction.
Qed.

(* The binders of two parts. *)
Lemma binders_not_free_app (xs xs' : list variable)
    (ks ks' : list continuation) (h : continuation_handler) :
  binders_not_free (xs ++ xs') (ks ++ ks') h
  = binders_not_free xs ks h && binders_not_free xs' ks' h.
Proof.
  unfold binders_not_free. rewrite !forallb_app.
  destruct (forallb (fun x => negb (var_free_in_handler_b x h)) xs),
    (forallb (fun x => negb (var_free_in_handler_b x h)) xs'),
    (forallb (fun k => negb (cont_free_in_handler_b k h)) ks),
    (forallb (fun k => negb (cont_free_in_handler_b k h)) ks');
    reflexivity.
Qed.

(* As [find_use_refines_aux], with one case more: a [Let_cont] whose
   handler applies the continuation. The handler of the continuation
   is copied to the head of that handler, where the applications of
   that handler are gathered and the copy is rewritten; the original
   then passes that [Let_cont], whose handler no longer mentions it,
   and is rewritten for the applications that are left; and the copy
   of the handler as rewritten goes back. *)
Fixpoint gather_uses_refines_aux (body : expr) {struct body} :
  ∀ (tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' : expr)
    (target : Typing_env.t) (uses : list (Typing_env.t * list simple))
    (xs : list variable) (ks : list continuation),
    gather_uses tenv k (Handler params metadata handler) body
    = Some (uses, xs, ks) ->
    binders_not_free xs ks (Handler params metadata handler') = true ->
    (∀ use : Typing_env.t * list simple,
      use ∈ uses ->
      application_admits params metadata (fst use) (snd use) target) ->
    expr_refines_under target handler handler' ->
    expr_refines_under tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  destruct body
    as [x declared d rest | bindings rest | k' h' rest | handlers rest | ap
        | ac | scrutinee arms | ];
    intros tenv k params metadata handler handler' target uses xs ks Hgather
      Hfree Huses Hhandler;
    cbn [gather_uses] in Hgather; try discriminate.
  - destruct (var_free_in_handler_b x (Handler params metadata handler))
      eqn:Hx; [ discriminate | ].
    destruct (let_env tenv x declared d) as [tenv' | ] eqn:Henv;
      [ | discriminate ].
    destruct (gather_uses tenv' k (Handler params metadata handler) rest)
      as [[[uses' xs'] ks'] | ] eqn:Hrest; [ | discriminate ].
    injection Hgather as <- <- <-.
    unfold binders_not_free in Hfree. cbn [forallb] in Hfree.
    apply andb_true_iff in Hfree. destruct Hfree as [Hxs Hks].
    apply andb_true_iff in Hxs. destruct Hxs as [Hx' Hxs].
    apply negb_true_iff in Hx'.
    assert (Hrest_free :
      binders_not_free xs' ks' (Handler params metadata handler') = true).
    { unfold binders_not_free. rewrite Hxs, Hks. reflexivity. }
    pose proof (gather_uses_refines_aux rest tenv' k params metadata handler
                  handler' target uses' xs' ks' Hrest Hrest_free Huses
                  Hhandler)
      as IH.
    intros vars conts statics hp rd Hsat.
    apply (refines_trans _
             (State
                (Let x declared d
                   (Let_cont_non_recursive k
                      (Handler params metadata handler) rest))
                vars conts statics hp rd)).
    { apply equivalent_refines, let_cont_past_let. intros Hfree'.
      apply var_free_in_handler_b_sound in Hfree'. congruence. }
    apply (refines_trans _
             (State
                (Let x declared d
                   (Let_cont_non_recursive k
                      (Handler params metadata handler') rest))
                vars conts statics hp rd)).
    { apply let_refines_step_exec; [ reflexivity | intros v hp' rd' Hv Hkind ].
      apply IH.
      exact (let_env_satisfied tenv tenv' x declared d statics hp hp' rd rd'
               vars v Henv Hsat Hv Hkind). }
    apply equivalent_refines, equivalent_sym, let_cont_past_let.
    intros Hfree'. apply var_free_in_handler_b_sound in Hfree'. congruence.
  - destruct (negb (Pos.eqb k k')
              && negb
                   (cont_free_in_handler_b k'
                      (Handler params metadata handler)))
      eqn:Htests; [ | discriminate ].
    apply andb_true_iff in Htests. destruct Htests as [Hne Hk'].
    apply negb_true_iff in Hne, Hk'. apply Pos.eqb_neq in Hne.
    destruct (gather_uses tenv k (Handler params metadata handler) rest)
      as [[[uses_rest xs_rest] ks_rest] | ] eqn:Hrest; [ | discriminate ].
    destruct (cont_free_in_handler_b k h') eqn:Hk.
    + (* the handler applies the continuation: a copy goes into it *)
      destruct h' as [params' metadata' body'].
      destruct (negb
                  (cont_free_in_handler_b k
                     (Handler params metadata handler))
                && forallb
                     (fun x =>
                        negb
                          (var_free_in_handler_b x
                             (Handler params metadata handler)))
                     params')
        eqn:Htests'; [ | discriminate ].
      apply andb_true_iff in Htests'. destruct Htests' as [Hself Hparams].
      apply negb_true_iff in Hself.
      destruct (handler_env tenv params' metadata') as [tenv_handler | ]
        eqn:Henv; [ | discriminate ].
      destruct (gather_uses tenv_handler k (Handler params metadata handler)
                  body')
        as [[[uses_h xs_h] ks_h] | ] eqn:Hinner; [ | discriminate ].
      injection Hgather as <- <- <-.
      change (k :: k' :: ks_h ++ ks_rest)
        with ([k; k'] ++ ks_h ++ ks_rest) in Hfree.
      rewrite binders_not_free_app, binders_not_free_app in Hfree.
      apply andb_true_iff in Hfree. destruct Hfree as [Hfree_head Hfree].
      apply andb_true_iff in Hfree. destruct Hfree as [Hfree_h Hfree_rest].
      unfold binders_not_free in Hfree_head. cbn [forallb] in Hfree_head.
      apply andb_true_iff in Hfree_head.
      destruct Hfree_head as [Hparams' Hconts'].
      rewrite andb_true_r in Hconts'.
      apply andb_true_iff in Hconts'. destruct Hconts' as [Hself' Hk''].
      apply negb_true_iff in Hself', Hk''.
      pose proof (gather_uses_refines_aux body' tenv_handler k params
                    metadata handler handler' target uses_h xs_h ks_h Hinner
                    Hfree_h
                    (fun use Hin =>
                       Huses use (in_or_app _ _ _ (or_introl Hin)))
                    Hhandler)
        as IHh.
      pose proof (gather_uses_refines_aux rest tenv k params metadata
                    handler handler' target uses_rest xs_rest ks_rest Hrest
                    Hfree_rest
                    (fun use Hin =>
                       Huses use (in_or_app _ _ _ (or_intror Hin)))
                    Hhandler)
        as IHrest.
      set (h := Handler params metadata handler) in *.
      set (hr := Handler params metadata handler') in *.
      assert (Hnot_self : ~ cont_free_in_handler k h).
      { intros Hf. apply cont_free_in_handler_b_sound in Hf. congruence. }
      assert (Hnot_self' : ~ cont_free_in_handler k hr).
      { intros Hf. apply cont_free_in_handler_b_sound in Hf. congruence. }
      assert (Hnot_params :
        ∀ param : variable, param ∈ params' -> ~ var_free_in_handler param h).
      { intros param Hin Hf. apply var_free_in_handler_b_sound in Hf.
        rewrite forallb_forall in Hparams. specialize (Hparams param Hin).
        rewrite Hf in Hparams. discriminate. }
      assert (Hnot_params' :
        ∀ param : variable,
          param ∈ params' -> ~ var_free_in_handler param hr).
      { intros param Hin Hf. apply var_free_in_handler_b_sound in Hf.
        rewrite forallb_forall in Hparams'. specialize (Hparams' param Hin).
        rewrite Hf in Hparams'. discriminate. }
      intros vars conts statics hp rd Hsat.
      (* the copy *)
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k h
                     (Let_cont_non_recursive k'
                        (Handler params' metadata'
                           (Let_cont_non_recursive k h body'))
                        rest))
                  vars conts statics hp rd)).
      { apply equivalent_refines.
        exact (let_cont_copied_into_handler k k' h params' metadata' body'
                 rest Hnot_self Hnot_params vars conts statics hp rd). }
      (* the copy rewritten, in the handler *)
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k h
                     (Let_cont_non_recursive k'
                        (Handler params' metadata'
                           (Let_cont_non_recursive k hr body'))
                        rest))
                  vars conts statics hp rd)).
      { apply (let_cont_body_refines_under tenv k h _ _); [ | exact Hsat ].
        exact (let_cont_handler_typed tenv k' params' metadata' _ _ rest
                 tenv_handler Henv IHh). }
      (* the original past the other [Let_cont], which no longer uses it *)
      assert (Hbound :
        ~ cont_free_in_handler k
            (Handler params' metadata' (Let_cont_non_recursive k hr body'))).
      { simpl. tauto. }
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k'
                     (Handler params' metadata'
                        (Let_cont_non_recursive k hr body'))
                     (Let_cont_non_recursive k h rest))
                  vars conts statics hp rd)).
      { apply equivalent_refines, let_cont_past_let_cont;
          [ exact Hne | exact Hbound | ].
        intros Hf. apply cont_free_in_handler_b_sound in Hf. congruence. }
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k'
                     (Handler params' metadata'
                        (Let_cont_non_recursive k hr body'))
                     (Let_cont_non_recursive k hr rest))
                  vars conts statics hp rd)).
      { exact (let_cont_body_refines_under tenv k' _ _ _ IHrest vars conts
                 statics hp rd Hsat). }
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k hr
                     (Let_cont_non_recursive k'
                        (Handler params' metadata'
                           (Let_cont_non_recursive k hr body'))
                        rest))
                  vars conts statics hp rd)).
      { apply equivalent_refines, equivalent_sym, let_cont_past_let_cont;
          [ exact Hne | exact Hbound | ].
        intros Hf. apply cont_free_in_handler_b_sound in Hf. congruence. }
      apply equivalent_refines, equivalent_sym.
      exact (let_cont_copied_into_handler k k' hr params' metadata' body'
               rest Hnot_self' Hnot_params' vars conts statics hp rd).
    + injection Hgather as <- <- <-.
      unfold binders_not_free in Hfree. cbn [forallb] in Hfree.
      apply andb_true_iff in Hfree. destruct Hfree as [Hxs Hks].
      apply andb_true_iff in Hks. destruct Hks as [Hk'' Hks].
      apply negb_true_iff in Hk''.
      assert (Hrest_free :
        binders_not_free xs_rest ks_rest (Handler params metadata handler')
        = true).
      { unfold binders_not_free. rewrite Hxs, Hks. reflexivity. }
      pose proof (gather_uses_refines_aux rest tenv k params metadata
                    handler handler' target uses_rest xs_rest ks_rest Hrest
                    Hrest_free Huses Hhandler)
        as IH.
      intros vars conts statics hp rd Hsat.
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k' h'
                     (Let_cont_non_recursive k
                        (Handler params metadata handler) rest))
                  vars conts statics hp rd)).
      { apply equivalent_refines, let_cont_past_let_cont;
          [ exact Hne | | ];
          intros Hfree'; apply cont_free_in_handler_b_sound in Hfree';
          congruence. }
      apply (refines_trans _
               (State
                  (Let_cont_non_recursive k' h'
                     (Let_cont_non_recursive k
                        (Handler params metadata handler') rest))
                  vars conts statics hp rd)).
      { exact (let_cont_body_refines_under tenv k' h' _ _ IH vars conts
                 statics hp rd Hsat). }
      apply equivalent_refines, equivalent_sym, let_cont_past_let_cont;
        [ exact Hne | | ];
        intros Hfree'; apply cont_free_in_handler_b_sound in Hfree';
        congruence.
  - destruct (leaf_uses tenv k (Apply_cont ac)) as [uses' | ] eqn:Hleaf;
      [ | discriminate ].
    injection Hgather as <- _ _.
    apply (leaf_refines tenv k params metadata handler handler' _ uses'
             Hleaf).
    intros index tenv_use args statics hp rd valuation vars conts vs
      handler_vars Hnth _ Hshows Hargs Hkinds Hbind.
    apply Hhandler.
    exact (Huses (tenv_use, args) (nth_error_In _ _ Hnth) statics vars vs
             handler_vars
             (shows_satisfies tenv_use statics valuation vars Hshows) Hargs
             Hkinds Hbind).
  - destruct (leaf_uses tenv k (Switch scrutinee arms)) as [uses' | ]
      eqn:Hleaf; [ | discriminate ].
    injection Hgather as <- _ _.
    apply (leaf_refines tenv k params metadata handler handler' _ uses'
             Hleaf).
    intros index tenv_use args statics hp rd valuation vars conts vs
      handler_vars Hnth _ Hshows Hargs Hkinds Hbind.
    apply Hhandler.
    exact (Huses (tenv_use, args) (nth_error_In _ _ Hnth) statics vars vs
             handler_vars
             (shows_satisfies tenv_use statics valuation vars Hshows) Hargs
             Hkinds Hbind).
Qed.

Theorem gather_uses_refines : gather_uses_refines_statement.
Proof.
  intros tenv k params metadata handler handler' body target uses xs ks.
  exact (gather_uses_refines_aux body tenv k params metadata handler
           handler' target uses xs ks).
Qed.

(* The one application gathered: the environment that [use_env] makes
   for it is one that it admits. *)
Lemma find_gathered_use_refines (tenv : Typing_env.t) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr) (target : Typing_env.t)
    (xs : list variable) (ks : list continuation) :
  find_gathered_use tenv k (Handler params metadata handler) body
  = Some (target, xs, ks) ->
  binders_not_free xs ks (Handler params metadata handler') = true ->
  expr_refines_under target handler handler' ->
  expr_refines_under tenv
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  unfold find_gathered_use. intros Hfind Hfree Hhandler.
  destruct (gather_uses tenv k (Handler params metadata handler) body)
    as [[[uses xs'] ks'] | ] eqn:Hgather; [ | discriminate ].
  destruct uses as [ | [tenv_use args] [ | second others]];
    try discriminate.
  destruct (use_env tenv tenv_use params metadata args) as [target' | ]
    eqn:Huse; [ | discriminate ].
  injection Hfind as <- <- <-.
  apply (gather_uses_refines_aux body tenv k params metadata handler
           handler' target' [(tenv_use, args)] xs' ks' Hgather Hfree);
    [ | exact Hhandler ].
  intros use [<- | []]. cbn [fst snd].
  intros statics vars vs handler_vars Hsat Hargs Hkinds Hbind.
  exact (use_env_satisfied tenv tenv_use params metadata args target'
           statics vars vs handler_vars Huse Hsat Hargs Hkinds Hbind).
Qed.

(* [find_use], and where it finds none, the one gathered. *)
Lemma find_one_use_refines (tenv : Typing_env.t) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr) (target : Typing_env.t)
    (xs : list variable) (ks : list continuation) :
  find_one_use tenv k (Handler params metadata handler) body
  = Some (target, xs, ks) ->
  binders_not_free xs ks (Handler params metadata handler') = true ->
  expr_refines_under target handler handler' ->
  expr_refines_under tenv
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  unfold find_one_use. intros Hfind Hfree Hhandler.
  destruct (find_use tenv tenv k (Handler params metadata handler) body)
    as [found | ] eqn:Huse.
  - injection Hfind as ->.
    exact (find_use_refines_aux body tenv tenv k params metadata handler
             handler' target xs ks Huse Hfree Hhandler).
  - exact (find_gathered_use_refines tenv k params metadata handler
             handler' body target xs ks Hfind Hfree Hhandler).
Qed.

(* By recursion on the term: the bodies of the code that a [Let_static]
   binds are inside its list of bindings, which the induction principle
   of terms does not go into. The bodies are rewritten one binding at a
   time. *)
Fixpoint simplify_expr_sound (e : expr) {struct e} :
  ∀ tenv : Typing_env.t, expr_refines_under tenv e (simplify_expr tenv e).
Proof.
  destruct e
    as [x declared d body | bindings body | k handler body
        | handlers body | ap | ac | scrutinee arms | ];
    intros tenv; cbn [simplify_expr].
  - pose proof (resolve_value_slot_refines tenv x declared d body)
      as Hresolve.
    pose proof (simplify_let_sound tenv x (resolve_value_slot tenv declared d))
      as Hlet.
    destruct (simplify_let tenv x (resolve_value_slot tenv declared d))
      as [ | d' tenv' | ].
    + eapply expr_refines_under_trans; [ exact Hresolve | ].
      apply expr_equivalent_under_refines. exact (Hlet declared body).
    + eapply expr_refines_under_trans; [ exact Hresolve | ].
      exact (Hlet declared body (simplify_expr tenv' body)
               (simplify_expr_sound body tenv')).
    + destruct (declare_variable tenv x declared) as [tenv' | ] eqn:Hdeclare;
        [ | apply expr_refines_under_refl ].
      revert Hdeclare. unfold declare_variable.
      destruct (typing_env_well_formed tenv
                && negb (Typing_env.mem None tenv (Name_var x))) eqn:Htests;
        [ | discriminate ].
      intros [= <-]. apply andb_true_iff in Htests.
      destruct Htests as [Hwf Hfresh]. apply negb_true_iff in Hfresh.
      apply (let_declared_refines_under tenv x declared d body _ Hwf Hfresh).
      apply simplify_expr_sound.
  - cbv zeta.
    assert (Hbindings :
      ∀ (tenv' : Typing_env.t) (before : list static_binding),
        add_static_bindings tenv (before ++ bindings) = tenv' ->
        expr_refines_under tenv (Let_static (before ++ bindings) body)
          (Let_static
             (before ++ map (simplify_static_binding tenv') bindings) body)).
    { intros tenv'.
      induction bindings as [ | binding bindings IH]; intros before Henv;
        [ apply expr_refines_under_refl | ].
      cbn [map].
      apply (expr_refines_under_trans tenv _
               (Let_static
                  (before ++ simplify_static_binding tenv' binding :: bindings)
                  body)
               _).
      - destruct binding as [id c | id | sym const];
          cbn [simplify_static_binding]; try apply expr_refines_under_refl.
        destruct c as [params my_closure ret exn metadata code_body].
        cbn [simplify_code].
        destruct (code_env_in tenv' params my_closure metadata)
          as [tenv_code | ] eqn:Hcode; [ | apply expr_refines_under_refl ].
        cbv zeta.
        destruct (binds_statics code_body) eqn:Hsource; cbn [negb andb];
          [ apply expr_refines_under_refl | ].
        destruct (binds_statics (simplify_expr tenv_code code_body))
          eqn:Htarget; cbn [negb];
          [ apply expr_refines_under_refl | ].
        apply (let_static_code_typed tenv before id params my_closure ret exn
                 metadata code_body _ bindings body tenv_code);
          [ rewrite Henv; exact Hcode | exact Hsource | exact Htarget | ].
        apply (simplify_expr_sound code_body tenv_code).
      - specialize
          (IH (before ++ [simplify_static_binding tenv' binding])).
        rewrite <- !app_assoc in IH. cbn [app] in IH. apply IH.
        rewrite <- Henv.
        destruct binding as [id c | id | sym const];
          cbn [simplify_static_binding]; try reflexivity.
        apply add_static_bindings_code. }
    eapply expr_refines_under_trans.
    + exact (Hbindings (add_static_bindings tenv bindings) [] eq_refl).
    + apply let_static_typed. rewrite add_static_bindings_simplify.
      apply simplify_expr_sound.
  - cbv zeta. eapply expr_refines_under_trans.
    + apply let_cont_body_refines_under. apply (simplify_expr_sound body).
    + destruct handler as [params metadata handler].
      generalize (simplify_expr tenv body). intros body'.
      assert (Hdefault :
        expr_refines_under tenv
          (Let_cont_non_recursive k (Handler params metadata handler) body')
          (Let_cont_non_recursive k
             (simplify_handler false tenv (Handler params metadata handler))
             body')).
      { cbn [simplify_handler].
        destruct (handler_env tenv params metadata) as [tenv_handler | ]
          eqn:Henv; [ | apply expr_refines_under_refl ].
        cbv zeta. cbn [negb orb].
        exact (let_cont_handler_typed tenv k params metadata handler _ _
                 tenv_handler Henv
                 (simplify_expr_sound handler tenv_handler)). }
      destruct (find_one_use tenv k (Handler params metadata handler) body')
        as [[[target xs] ks] | ] eqn:Hfind; [ | exact Hdefault ].
      destruct (binders_not_free xs ks
                  (Handler params metadata (simplify_expr target handler)))
        eqn:Hfree; [ | exact Hdefault ].
      exact (find_one_use_refines tenv k params metadata handler _ body'
               target xs ks Hfind Hfree
               (simplify_expr_sound handler target)).
  - eapply expr_refines_under_trans.
    + apply let_cont_recursive_body_refines_under.
      apply (simplify_expr_sound body).
    + generalize (simplify_expr tenv body). intros body'.
      assert (Hhandlers :
        ∀ before : list (continuation * continuation_handler),
          expr_refines_under tenv
            (Let_cont_recursive (before ++ handlers) body')
            (Let_cont_recursive
               (before
                ++ map (fun '(k, h) => (k, simplify_handler true tenv h))
                     handlers)
               body')).
      { induction handlers as [ | [k h] handlers IH]; intros before;
          [ apply expr_refines_under_refl | ].
        cbn [map].
        apply (expr_refines_under_trans tenv _
                 (Let_cont_recursive
                    (before ++ (k, simplify_handler true tenv h) :: handlers)
                    body')
                 _).
        - destruct h as [params metadata handler]. cbn [simplify_handler].
          destruct (handler_env tenv params metadata) as [tenv_handler | ]
            eqn:Henv; [ | apply expr_refines_under_refl ].
          cbv zeta. cbn [negb orb].
          destruct (Bool.eqb (binds_statics handler)
                      (binds_statics (simplify_expr tenv_handler handler)))
            eqn:Hbinds; [ | apply expr_refines_under_refl ].
          apply (let_cont_recursive_handler_typed tenv before k params
                   metadata handler _ handlers body' tenv_handler Henv);
            [ apply eqb_prop; exact Hbinds
            | apply (simplify_expr_sound handler tenv_handler) ].
        - specialize (IH (before ++ [(k, simplify_handler true tenv h)])).
          rewrite <- !app_assoc in IH. exact IH. }
      exact (Hhandlers []).
  - apply simplify_apply_refines_under.
  - apply expr_equivalent_under_refines. apply simplify_apply_cont_equivalent.
  - apply simplify_switch_refines.
  - apply expr_refines_under_refl.
Qed.

Theorem simplify_expr_refines_under : simplify_expr_refines_under_statement.
Proof. intros tenv e. apply simplify_expr_sound. Qed.

Theorem rewrite_handler_at_use_refines
    : rewrite_handler_at_use_refines_statement.
Proof.
  intros tenv k [params metadata handler] h' body.
  unfold rewrite_handler_at_use.
  destruct (find_use tenv tenv k (Handler params metadata handler) body)
    as [[[target xs] ks] | ] eqn:Hfind; [ | discriminate ].
  cbv zeta.
  destruct (binders_not_free xs ks
              (Handler params metadata (simplify_expr target handler)))
    eqn:Hfree; [ | discriminate ].
  intros [= <-].
  exact (find_use_refines tenv tenv k params metadata handler _ body target
           xs ks Hfind Hfree (simplify_expr_sound handler target)).
Qed.

Theorem simplify_by_types_refines : simplify_by_types_refines_statement.
Proof.
  intros e. apply expr_refines_under_create.
  exact (simplify_expr_refines_under Typing_env.create e).
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Theorem example_pass_add_constants : example_pass_add_constants_statement.
Proof. intros k. vm_compute. reflexivity. Qed.

Theorem example_pass_is_int : example_pass_is_int_statement.
Proof. intros k0 k1. vm_compute. reflexivity. Qed.

Theorem example_pass_second_of_pair : example_pass_second_of_pair_statement.
Proof. intros k. vm_compute. reflexivity. Qed.

Theorem example_pass_add_zero : example_pass_add_zero_statement.
Proof. intros k. vm_compute. reflexivity. Qed.

Theorem example_pass_unbox_of_box : example_pass_unbox_of_box_statement.
Proof. intros k. vm_compute. reflexivity. Qed.

Theorem example_pass_untag_of_tag : example_pass_untag_of_tag_statement.
Proof. intros k. vm_compute. reflexivity. Qed.

Theorem example_pass_under_binders : example_pass_under_binders_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_pass_in_code : example_pass_in_code_statement.
Proof. intros k ret exn. vm_compute. reflexivity. Qed.

Theorem example_pass_unbox_in_code : example_pass_unbox_in_code_statement.
Proof. intros k ret exn. vm_compute. reflexivity. Qed.

Theorem example_pass_phys_equal : example_pass_phys_equal_statement.
Proof. intros k0 k1. vm_compute. reflexivity. Qed.

Theorem example_pass_call : example_pass_call_statement.
Proof. intros k exn. vm_compute. reflexivity. Qed.

Theorem example_pass_call_from_pair : example_pass_call_from_pair_statement.
Proof. intros k exn. vm_compute. reflexivity. Qed.

Theorem example_pass_value_slot : example_pass_value_slot_statement.
Proof. intros k. vm_compute. reflexivity. Qed.

Theorem example_pass_static_block : example_pass_static_block_statement.
Proof. intros k. vm_compute. reflexivity. Qed.

Theorem example_pass_unknown_let : example_pass_unknown_let_statement.
Proof. intros k. vm_compute. reflexivity. Qed.

Theorem example_pass_static_in_code
    : example_pass_static_in_code_statement.
Proof. intros k ret exn. vm_compute. reflexivity. Qed.

Theorem example_pass_static_closure
    : example_pass_static_closure_statement.
Proof. intros k exn ret exn'. vm_compute. reflexivity. Qed.

Theorem example_pass_loop : example_pass_loop_statement.
Proof. intros k. vm_compute. reflexivity. Qed.

Theorem example_pass_heap_primitive
    : example_pass_heap_primitive_statement.
Proof. intros k. vm_compute. reflexivity. Qed.

Theorem example_handler_at_use : example_handler_at_use_statement.
Proof. vm_compute. reflexivity. Qed.
