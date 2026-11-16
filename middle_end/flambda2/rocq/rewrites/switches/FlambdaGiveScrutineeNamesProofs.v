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
  FlambdaUnaliasParamsNamesProofs FlambdaGiveScrutineeSpec.
Import ListNotations.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

Lemma first_arm_of_equivariant (r : renaming) (k : continuation)
    (arms : list (Z * apply_cont)) :
  first_arm_of (rename_continuation r k)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = first_arm_of k arms.
Proof.
  unfold first_arm_of.
  induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
  cbn [map first_answer fst snd rename_apply_cont ac_continuation].
  rewrite rename_continuation_eqb.
  destruct (Pos.eqb (ac_continuation ac) k); [ reflexivity | exact IH ].
Qed.

Lemma only_arm_equivariant (r : renaming) (k : continuation) (c : Z)
    (arms : list (Z * apply_cont)) :
  only_arm (rename_continuation r k) c
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = only_arm k c arms.
Proof.
  unfold only_arm.
  induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
  cbn [map forallb fst snd rename_apply_cont ac_continuation].
  rewrite rename_continuation_eqb, IH. reflexivity.
Qed.

Fixpoint arm_fact_equivariant (r : renaming) (x : variable)
    (k : continuation) (body : expr) {struct body} :
  arm_fact (rename_variable r x) (rename_continuation r k)
    (rename_expr r body)
  = arm_fact x k body.
Proof.
  destruct body as [ | | k' hh body | | | | scrutinee arms | ];
    try reflexivity.
  - cbn [rename_expr arm_fact].
    rewrite rename_continuation_eqb, cont_free_in_handler_b_equivariant,
      arm_fact_equivariant.
    reflexivity.
  - cbn [rename_expr arm_fact].
    destruct scrutinee as [y | sym | cst]; cbn [rename_simple];
      try reflexivity.
    rewrite rename_variable_eqb, first_arm_of_equivariant.
    destruct (Pos.eqb y x); [ | reflexivity ].
    destruct (first_arm_of k arms) as [c | ]; [ | reflexivity ].
    rewrite only_arm_equivariant. reflexivity.
Qed.

Lemma arm_known_equivariant (r : renaming) (x : variable)
    (scrutinee : simple) (d : Z) (known : option Z) :
  arm_known (rename_variable r x) (rename_simple r scrutinee) d known
  = arm_known x scrutinee d known.
Proof.
  destruct scrutinee as [y | sym | cst]; cbn [rename_simple arm_known];
    try reflexivity.
  rewrite rename_variable_eqb. reflexivity.
Qed.

Lemma known_under_equivariant (r : renaming) (x : variable)
    (params : list variable) (known : option Z) :
  known_under (rename_variable r x) (map (rename_variable r) params) known
  = known_under x params known.
Proof. unfold known_under. rewrite among_rename_variable. reflexivity. Qed.

Lemma give_nth_equivariant (r : renaming) (x : variable) (c : Z) :
  ∀ (i : nat) (args : list simple),
    give_nth (rename_variable r x) c i (map (rename_simple r) args)
    = map (rename_simple r) (give_nth x c i args).
Proof.
  induction i as [ | i IH]; intros [ | a args]; try reflexivity.
  - destruct a as [y | sym | c0]; cbn [map rename_simple give_nth];
      try reflexivity.
    destruct c0; try reflexivity.
    destruct (Z.eqb _ c); reflexivity.
  - cbn [map give_nth]. rewrite IH. reflexivity.
Qed.

Lemma give_in_apply_cont_equivariant (r : renaming) (x : variable)
    (k : continuation) (i : nat) (known : option Z) (ac : apply_cont) :
  give_in_apply_cont (rename_variable r x) (rename_continuation r k) i known
    (rename_apply_cont r ac)
  = rename_apply_cont r (give_in_apply_cont x k i known ac).
Proof.
  unfold give_in_apply_cont. destruct known as [c | ]; [ | reflexivity ].
  unfold rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action].
  rewrite rename_continuation_eqb.
  destruct (Pos.eqb (ac_continuation ac) k); [ | reflexivity ].
  cbn [ac_continuation ac_args ac_trap_action].
  rewrite give_nth_equivariant. reflexivity.
Qed.

Fixpoint give_scrutinee_equivariant (r : renaming) (x : variable)
    (k : continuation) (i : nat) (known : option Z) (e : expr) {struct e} :
  give_scrutinee (rename_variable r x) (rename_continuation r k) i known
    (rename_expr r e)
  = rename_expr r (give_scrutinee x k i known e)
with give_in_handler_equivariant (r : renaming) (x : variable)
    (k : continuation) (i : nat) (known : option Z)
    (h : continuation_handler) {struct h} :
  give_in_handler (rename_variable r x) (rename_continuation r k) i known
    (rename_handler r h)
  = rename_handler r (give_in_handler x k i known h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr give_scrutinee]; try reflexivity.
    + rewrite rename_variable_eqb, give_scrutinee_equivariant. reflexivity.
    + rewrite give_scrutinee_equivariant. reflexivity.
    + rewrite arm_fact_equivariant, give_in_handler_equivariant,
        give_scrutinee_equivariant.
      reflexivity.
    + rewrite give_scrutinee_equivariant. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map]. rewrite give_in_handler_equivariant, IH. reflexivity.
    + rewrite give_in_apply_cont_equivariant. reflexivity.
    + f_equal. induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
      cbn [map fst snd].
      rewrite arm_known_equivariant, give_in_apply_cont_equivariant, IH.
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler give_in_handler].
    rewrite known_under_equivariant, give_scrutinee_equivariant.
    reflexivity.
Qed.

Lemma scrutinee_by_equivariant (r : renaming) (k : continuation) (i : nat)
    (current : list variable) (ac : apply_cont) :
  scrutinee_by (rename_continuation r k) i
    (map (rename_variable r) current) (rename_apply_cont r ac)
  = option_map (map (rename_variable r)) (scrutinee_by k i current ac).
Proof.
  unfold scrutinee_by, rename_apply_cont.
  cbn [ac_continuation ac_args].
  rewrite rename_continuation_eqb, nth_error_map.
  destruct (Pos.eqb (ac_continuation ac) k); [ | reflexivity ].
  destruct (nth_error (ac_args ac) i) as [[y | sym | c] | ];
    try reflexivity.
  destruct c; reflexivity.
Qed.

Lemma scrutinee_var_equivariant (r : renaming) (current : list variable)
    (scrutinee : simple) :
  scrutinee_var (map (rename_variable r) current)
    (rename_simple r scrutinee)
  = map (rename_variable r) (scrutinee_var current scrutinee).
Proof. destruct scrutinee; reflexivity. Qed.

Fixpoint scrutinee_of_equivariant (r : renaming) (current : list variable)
    (e : expr) {struct e} :
  scrutinee_of (map (rename_variable r) current) (rename_expr r e)
  = map (rename_variable r) (scrutinee_of current e).
Proof.
  destruct e as [ | | k' hh body | | | | scrutinee arms | ];
    try reflexivity.
  - cbn [rename_expr scrutinee_of]. apply scrutinee_of_equivariant.
  - cbn [rename_expr scrutinee_of]. apply scrutinee_var_equivariant.
Qed.

Fixpoint first_scrutinee_equivariant (r : renaming) (k : continuation)
    (i : nat) (current : list variable) (e : expr) {struct e} :
  first_scrutinee (rename_continuation r k) i
    (map (rename_variable r) current) (rename_expr r e)
  = option_map (map (rename_variable r)) (first_scrutinee k i current e)
with handler_first_scrutinee_equivariant (r : renaming) (k : continuation)
    (i : nat) (current : list variable) (h : continuation_handler)
    {struct h} :
  handler_first_scrutinee (rename_continuation r k) i
    (map (rename_variable r) current) (rename_handler r h)
  = option_map (map (rename_variable r))
      (handler_first_scrutinee k i current h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr first_scrutinee]; try reflexivity.
    + apply first_scrutinee_equivariant.
    + apply first_scrutinee_equivariant.
    + rewrite scrutinee_of_equivariant, handler_first_scrutinee_equivariant,
        first_scrutinee_equivariant.
      destruct (handler_first_scrutinee k i (scrutinee_of current body) hh);
        reflexivity.
    + rewrite first_scrutinee_equivariant.
      assert (Hhandlers :
        first_answer
          (fun '(_, h) =>
             handler_first_scrutinee (rename_continuation r k) i
               (map (rename_variable r) current) h)
          (map (fun '(k0, h) => (rename_continuation r k0, rename_handler r h))
             handlers)
        = option_map (map (rename_variable r))
            (first_answer
               (fun '(_, h) => handler_first_scrutinee k i current h)
               handlers)).
      { induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
        cbn [map first_answer].
        rewrite handler_first_scrutinee_equivariant.
        destruct (handler_first_scrutinee k i current hh);
          [ reflexivity | exact IH ]. }
      rewrite Hhandlers.
      destruct
        (first_answer (fun '(_, h) => handler_first_scrutinee k i current h)
           handlers);
        reflexivity.
    + apply scrutinee_by_equivariant.
    + rewrite scrutinee_var_equivariant.
      induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
      cbn [map first_answer fst snd]. rewrite scrutinee_by_equivariant.
      destruct (scrutinee_by k i (scrutinee_var current scrutinee) ac);
        [ reflexivity | exact IH ].
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_first_scrutinee].
    apply first_scrutinee_equivariant.
Qed.

Lemma give_first_equivariant (r : renaming) (xs : list variable) (i : nat)
    (k : continuation) (h : continuation_handler) (body : expr) :
  give_first (map (rename_variable r) xs) i (rename_continuation r k)
    (rename_handler r h) (rename_expr r body)
  = rename_expr r (give_first xs i k h body).
Proof.
  induction xs as [ | x xs IH]; cbn [map give_first]; [ reflexivity | ].
  rewrite give_scrutinee_equivariant, unalias_param_equivariant.
  destruct (unalias_param i k h (give_scrutinee x k i None body))
    as [[h' body'] | ];
    [ reflexivity | exact IH ].
Qed.

Lemma give_at_equivariant (r : renaming) (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr) :
  give_at i (rename_continuation r k) (rename_handler r h)
    (rename_expr r body)
  = rename_expr r (give_at i k h body).
Proof.
  unfold give_at.
  pose proof (first_scrutinee_equivariant r k i [] body) as Hfirst.
  cbn [map] in Hfirst. rewrite Hfirst.
  destruct (first_scrutinee k i [] body) as [xs | ];
    cbn [option_map]; [ | reflexivity ].
  apply give_first_equivariant.
Qed.

Lemma give_from_equivariant (r : renaming) (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    give_from i (rename_continuation r k) (rename_handler r h)
      (rename_expr r body)
    = rename_expr r (give_from i k h body).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [give_from];
    [ reflexivity | ].
  rewrite give_at_equivariant. apply IH.
Qed.

Fixpoint give_scrutinees_equivariant_aux (r : renaming) (e : expr) :
  give_scrutinees (rename_expr r e) = rename_expr r (give_scrutinees e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler give_scrutinees]; try reflexivity.
  - rewrite give_scrutinees_equivariant_aux. reflexivity.
  - rewrite give_scrutinees_equivariant_aux. reflexivity.
  - rewrite !give_scrutinees_equivariant_aux, length_map.
    pose proof
      (give_from_equivariant r (length params) k
         (Handler params metadata (give_scrutinees handler))
         (give_scrutinees body))
      as Hfrom.
    cbn [rename_handler] in Hfrom. rewrite Hfrom. reflexivity.
  - rewrite give_scrutinees_equivariant_aux. reflexivity.
Qed.

Lemma give_scrutinees_renamed : rewrite_equivariant give_scrutinees.
Proof. exact give_scrutinees_equivariant_aux. Qed.

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
