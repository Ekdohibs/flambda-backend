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


(* FlambdaGiveScrutineeSpec.give_scrutinees commutes with renaming and
   keeps scope: the lemmas from which
   FlambdaValidatorEquivarianceProofs.v and FlambdaStaticsScopeProofs.v
   prove the two statements. There is no statement here, and nothing
   here needs review.

   The pass compares names by equality only and makes none, so each of
   its functions gives, on a renamed term, the renamed result. For the
   scope: the rewrite puts the variable for a constant only where
   something is known of it, and something is known in an arm of a
   [Switch] on the variable, or in a handler bound beside such a
   [Switch], or under those with no binder of the variable between:
   the variable is free in the term. The continuations and the code
   are as they were. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaValidatorEquivarianceSpec FlambdaValidatorEquivarianceBaseProofs
  FlambdaScopedProofs FlambdaUnusedContsSpec
  FlambdaUnaliasParamsSpec FlambdaUnaliasParamsProofs
  FlambdaUnaliasParamsNamesProofs FlambdaEquivarianceTactic
  FlambdaGiveScrutineeSpec.
Import ListNotations.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

(* The laws are proved from the definitions alone ([equivariance], of
   FlambdaEquivarianceTactic.v): the tactic states and proves, as it
   goes, the laws of the functions that the pass is made of. What is
   stated here besides is what another file uses by name (the laws of
   [arm_fact], [arm_known] and [known_under]) and what the tactic does
   not find by itself: the pass looks for the scrutinee from an empty
   list of variables, which is not written as a renamed list, so the
   law of [first_scrutinee] is stated and then given at that list. *)

(* [first_answer] takes a function: the rule by which the tactic goes
   under it, with the two lemmas of FlambdaUnaliasParamsNamesProofs.v. *)
#[local] Ltac eqv_extra ::=
  match goal with
  | |- context [first_answer ?f' ?l] =>
    match goal with
    | |- context [first_answer ?f l] =>
      tryif constr_eq f f' then fail
      else (eqv_no_renaming f;
            erewrite (first_answer_ext_among f' f _ l);
            [ | eqv_enter l; solve [ eqv_core ] ])
    end
  end.

Lemma arm_fact_equivariant (r : renaming) (x : variable)
    (k : continuation) (body : expr) :
  arm_fact (rename_variable r x) (rename_continuation r k)
    (rename_expr r body)
  = arm_fact x k body.
Proof. equivariance. Qed.

Lemma arm_known_equivariant (r : renaming) (x : variable)
    (scrutinee : simple) (d : Z) (known : option Z) :
  arm_known (rename_variable r x) (rename_simple r scrutinee) d known
  = arm_known x scrutinee d known.
Proof. equivariance. Qed.

Lemma known_under_equivariant (r : renaming) (x : variable)
    (params : list variable) (known : option Z) :
  known_under (rename_variable r x) (map (rename_variable r) params) known
  = known_under x params known.
Proof. equivariance. Qed.

#[local] Hint Rewrite arm_fact_equivariant arm_known_equivariant
  known_under_equivariant : equivariance.

Lemma first_scrutinee_equivariant (r : renaming) (k : continuation)
    (i : nat) (current : list variable) (e : expr) :
  first_scrutinee (rename_continuation r k) i
    (map (rename_variable r) current) (rename_expr r e)
  = option_map (map (rename_variable r)) (first_scrutinee k i current e).
Proof. equivariance. Qed.

Lemma first_scrutinee_at_nil (r : renaming) (k : continuation) (i : nat)
    (e : expr) :
  first_scrutinee (rename_continuation r k) i [] (rename_expr r e)
  = option_map (map (rename_variable r)) (first_scrutinee k i [] e).
Proof. exact (first_scrutinee_equivariant r k i [] e). Qed.

#[local] Hint Rewrite first_scrutinee_at_nil : equivariance.

Lemma give_scrutinees_renamed : rewrite_equivariant give_scrutinees.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

Lemma give_in_apply_cont_continuations (x : variable) (k : continuation)
    (i : nat) (known : option Z) (ac : apply_cont) :
  apply_cont_continuations (give_in_apply_cont x k i known ac)
  = apply_cont_continuations ac.
Proof.
  unfold give_in_apply_cont.
  destruct known; [ destruct (Pos.eqb _ _) | ]; reflexivity.
Qed.

(* The variable of a fact is the scrutinee of a [Switch] that the body
   comes to, so it is free in the body. *)
Fixpoint arm_fact_free (x : variable) (k : continuation) (body : expr)
    (c : Z) {struct body} :
  arm_fact x k body = Some c -> In x (free_variables_of_expr body).
Proof.
  destruct body as [ | | k' hh body | | | | scrutinee arms | ];
    try discriminate.
  - cbn [arm_fact].
    destruct (Pos.eqb k' k || cont_free_in_handler_b k hh);
      [ discriminate | ].
    intros Hfact. cbn [free_variables_of_expr]. apply in_or_app. right.
    exact (arm_fact_free x k body c Hfact).
  - cbn [arm_fact]. destruct scrutinee as [z | | ]; try discriminate.
    destruct (Pos.eqb_spec z x) as [-> | _]; [ | discriminate ].
    intros _. cbn [free_variables_of_expr free_variables_of_simple app].
    left. reflexivity.
Qed.

Lemma give_nth_free_variables (x : variable) (c : Z) (a : variable) :
  ∀ (i : nat) (args : list simple),
    In a (free_variables_of_simples (give_nth x c i args)) ->
    In a (free_variables_of_simples args) ∨ a = x.
Proof.
  unfold free_variables_of_simples.
  induction i as [ | i IH]; intros [ | s args]; cbn [give_nth];
    try (intros Hin; left; exact Hin).
  - destruct s as [y | sym | c0]; try (intros Hin; left; exact Hin).
    destruct c0; try (intros Hin; left; exact Hin).
    destruct (Z.eqb _ c); [ | intros Hin; left; exact Hin ].
    cbn [flat_map free_variables_of_simple app].
    intros [<- | Hin]; [ right; reflexivity | left; exact Hin ].
  - cbn [flat_map]. intros Hin. apply in_app_or in Hin as [Hin | Hin].
    + left. apply in_or_app. left. exact Hin.
    + destruct (IH args Hin) as [Hin' | ->]; [ | right; reflexivity ].
      left. apply in_or_app. right. exact Hin'.
Qed.

Lemma give_in_apply_cont_free_variables (x : variable) (k : continuation)
    (i : nat) (known : option Z) (ac : apply_cont) (a : variable) :
  In a (free_variables_of_apply_cont (give_in_apply_cont x k i known ac)) ->
  In a (free_variables_of_apply_cont ac) ∨ (a = x ∧ known <> None).
Proof.
  unfold give_in_apply_cont, free_variables_of_apply_cont.
  destruct known as [c | ]; [ | intros Hin; left; exact Hin ].
  destruct (Pos.eqb _ _); [ | intros Hin; left; exact Hin ].
  cbn [ac_args]. intros Hin.
  destruct (give_nth_free_variables x c a i (ac_args ac) Hin) as [Hin' | ->];
    [ left; exact Hin' | right; split; [ reflexivity | discriminate ] ].
Qed.

Fixpoint give_scrutinee_code_closed (x : variable) (k : continuation)
    (i : nat) (known : option Z) (e : expr) {struct e} :
  code_closed (give_scrutinee x k i known e) = code_closed e
with give_in_handler_code_closed (x : variable) (k : continuation)
    (i : nat) (known : option Z) (h : continuation_handler) {struct h} :
  handler_code_closed (give_in_handler x k i known h) = handler_code_closed h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [give_scrutinee code_closed]; try reflexivity.
    + apply give_scrutinee_code_closed.
    + rewrite give_scrutinee_code_closed. reflexivity.
    + rewrite give_in_handler_code_closed, give_scrutinee_code_closed.
      reflexivity.
    + rewrite give_scrutinee_code_closed. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map forallb].
      rewrite give_in_handler_code_closed, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [give_in_handler handler_code_closed].
    apply give_scrutinee_code_closed.
Qed.

Fixpoint give_scrutinee_free_continuations (x : variable) (k : continuation)
    (i : nat) (known : option Z) (e : expr) {struct e} :
  free_continuations_of_expr (give_scrutinee x k i known e)
  = free_continuations_of_expr e
with give_in_handler_free_continuations (x : variable) (k : continuation)
    (i : nat) (known : option Z) (h : continuation_handler) {struct h} :
  free_continuations_of_handler (give_in_handler x k i known h)
  = free_continuations_of_handler h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [give_scrutinee free_continuations_of_expr]; try reflexivity.
    + apply give_scrutinee_free_continuations.
    + rewrite give_scrutinee_free_continuations. reflexivity.
    + rewrite give_in_handler_free_continuations,
        give_scrutinee_free_continuations.
      reflexivity.
    + rewrite give_scrutinee_free_continuations. f_equal.
      * induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
        cbn [map fst]. rewrite IH. reflexivity.
      * f_equal.
        induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
        cbn [map flat_map].
        rewrite give_in_handler_free_continuations, IH. reflexivity.
    + apply give_in_apply_cont_continuations.
    + induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
      cbn [map flat_map snd].
      rewrite give_in_apply_cont_continuations, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [give_in_handler free_continuations_of_handler].
    apply give_scrutinee_free_continuations.
Qed.

(* A free variable of the result is one of the term, or the variable
   given, where something is known of it. *)
Fixpoint give_scrutinee_free_variables (x : variable) (k : continuation)
    (i : nat) (a : variable) (e : expr) {struct e} :
  ∀ known : option Z,
    In a (free_variables_of_expr (give_scrutinee x k i known e)) ->
    In a (free_variables_of_expr e) ∨ (a = x ∧ known <> None)
with give_in_handler_free_variables (x : variable) (k : continuation)
    (i : nat) (a : variable) (h : continuation_handler) {struct h} :
  ∀ known : option Z,
    In a (free_variables_of_handler (give_in_handler x k i known h)) ->
    In a (free_variables_of_handler h) ∨ (a = x ∧ known <> None).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      intros known; cbn [give_scrutinee free_variables_of_expr];
      try (intros Hin; left; exact Hin).
    + intros Hin. apply in_app_or in Hin as [Hin | Hin];
        [ left; apply in_or_app; left; exact Hin | ].
      apply in_without in Hin as [Hin Hy].
      destruct (give_scrutinee_free_variables x k i a body _ Hin)
        as [Hin' | [-> Hknown]].
      * left. apply in_or_app. right. apply in_without. split; assumption.
      * right. split; [ reflexivity | ].
        destruct (Pos.eqb y x); [ contradiction | exact Hknown ].
    + intros Hin. apply in_app_or in Hin as [Hin | Hin];
        [ left; apply in_or_app; left; exact Hin | ].
      destruct (give_scrutinee_free_variables x k i a body known Hin)
        as [Hin' | Hx]; [ | right; exact Hx ].
      left. apply in_or_app. right. exact Hin'.
    + intros Hin. apply in_app_or in Hin as [Hin | Hin].
      * destruct (give_in_handler_free_variables x k i a hh _ Hin)
          as [Hin' | [-> Hknown]];
          [ left; apply in_or_app; left; exact Hin' | ].
        destruct (arm_fact x k' body) as [c | ] eqn:Hfact;
          [ | right; split; [ reflexivity | exact Hknown ] ].
        left. apply in_or_app. right.
        exact (arm_fact_free x k' body c Hfact).
      * destruct (give_scrutinee_free_variables x k i a body known Hin)
          as [Hin' | Hx]; [ | right; exact Hx ].
        left. apply in_or_app. right. exact Hin'.
    + intros Hin. apply in_app_or in Hin as [Hin | Hin].
      * assert (Hhandlers :
          In a (flat_map (fun '(_, h) => free_variables_of_handler h) handlers)
          ∨ (a = x ∧ known <> None)).
        { induction handlers as [ | [k' hh] handlers IH];
            cbn [map flat_map] in Hin |- *; [ contradiction | ].
          apply in_app_or in Hin as [Hin | Hin].
          - destruct (give_in_handler_free_variables x k i a hh known Hin)
              as [Hin' | Hx]; [ | right; exact Hx ].
            left. apply in_or_app. left. exact Hin'.
          - destruct (IH Hin) as [Hin' | Hx]; [ | right; exact Hx ].
            left. apply in_or_app. right. exact Hin'. }
        destruct Hhandlers as [Hin' | Hx]; [ | right; exact Hx ].
        left. apply in_or_app. left. exact Hin'.
      * destruct (give_scrutinee_free_variables x k i a body known Hin)
          as [Hin' | Hx]; [ | right; exact Hx ].
        left. apply in_or_app. right. exact Hin'.
    + apply give_in_apply_cont_free_variables.
    + intros Hin. apply in_app_or in Hin as [Hin | Hin];
        [ left; apply in_or_app; left; exact Hin | ].
      assert (Harms :
        In a (flat_map (fun arm => free_variables_of_apply_cont (snd arm)) arms)
        ∨ (a = x
           ∧ (In x (free_variables_of_simple scrutinee) ∨ known <> None))).
      { induction arms as [ | [d ac] arms IH];
          cbn [map flat_map snd] in Hin |- *; [ contradiction | ].
        apply in_app_or in Hin as [Hin | Hin].
        - destruct
            (give_in_apply_cont_free_variables x k i
               (arm_known x scrutinee d known) ac a Hin)
            as [Hin' | [-> Hknown]];
            [ left; apply in_or_app; left; exact Hin' | ].
          right. split; [ reflexivity | ].
          unfold arm_known in Hknown.
          destruct scrutinee as [z | | ]; try (right; exact Hknown).
          destruct (Pos.eqb_spec z x) as [-> | _];
            [ left; left; reflexivity | right; exact Hknown ].
        - destruct (IH Hin) as [Hin' | Hx]; [ | right; exact Hx ].
          left. apply in_or_app. right. exact Hin'. }
      destruct Harms as [Hin' | [-> [Hx | Hknown]]].
      * left. apply in_or_app. right. exact Hin'.
      * left. apply in_or_app. left. exact Hx.
      * right. split; [ reflexivity | exact Hknown ].
  - destruct h as [params metadata handler]. intros known.
    cbn [give_in_handler free_variables_of_handler]. intros Hin.
    apply in_without in Hin as [Hin Hparams].
    destruct (give_scrutinee_free_variables x k i a handler _ Hin)
      as [Hin' | [-> Hknown]].
    + left. apply in_without. split; assumption.
    + right. split; [ reflexivity | ].
      unfold known_under in Hknown.
      destruct (among x params); [ contradiction | exact Hknown ].
Qed.

Lemma give_scrutinee_scoped (x : variable) (k : continuation) (i : nat)
    (e : expr) :
  scoped e (give_scrutinee x k i None e).
Proof.
  intros Hclosed. rewrite give_scrutinee_code_closed.
  split; [ exact Hclosed | split ].
  - intros a Ha.
    destruct (give_scrutinee_free_variables x k i a e None Ha)
      as [Hin | [_ Hknown]]; [ exact Hin | contradiction ].
  - rewrite give_scrutinee_free_continuations. apply incl_refl.
Qed.

Lemma give_at_scoped (i : nat) (k : continuation) (h : continuation_handler)
    (body : expr) :
  scoped (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h (give_at i k h body)).
Proof.
  unfold give_at.
  destruct (first_scrutinee k i [] body) as [xs | ]; [ | apply scoped_refl ].
  induction xs as [ | x xs IH]; cbn [give_first]; [ apply scoped_refl | ].
  destruct (unalias_param i k h (give_scrutinee x k i None body));
    [ | exact IH ].
  apply scoped_let_cont;
    [ apply scoped_handler_refl | apply give_scrutinee_scoped ].
Qed.

Lemma give_from_scoped (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    scoped (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k h (give_from i k h body)).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [give_from];
    [ apply scoped_refl | ].
  exact (scoped_trans _ _ _ (give_at_scoped i k h body)
    (IH k h (give_at i k h body))).
Qed.

Fixpoint give_scrutinees_scoped (e : expr) : scoped e (give_scrutinees e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [give_scrutinees]; try apply scoped_refl.
  - apply scoped_let; [ apply incl_refl | apply give_scrutinees_scoped ].
  - apply scoped_let_static;
      [ apply scoped_bindings_refl | apply give_scrutinees_scoped ].
  - refine (scoped_trans _ _ _ _
      (give_from_scoped (length params) k
         (Handler params metadata (give_scrutinees handler))
         (give_scrutinees body))).
    apply scoped_let_cont;
      [ apply scoped_handler_intro, give_scrutinees_scoped
      | apply give_scrutinees_scoped ].
  - apply scoped_let_cont_recursive;
      [ reflexivity | apply scoped_handlers_refl
      | apply give_scrutinees_scoped ].
Qed.

Lemma give_scrutinees_scope : keeps_scope give_scrutinees.
Proof. apply keeps_scope_intro, give_scrutinees_scoped. Qed.
