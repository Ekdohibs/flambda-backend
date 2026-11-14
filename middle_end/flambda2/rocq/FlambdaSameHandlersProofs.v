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

(* Proofs of the statements of FlambdaSameHandlersSpec.v. Nothing here
   needs review.

   Down the body of the [Let_cont] that binds [k], the environments are
   those of that [Let_cont] but for the names bound on the way. A
   handler that passes the test is the same as that of [k] up to bound
   names and has none of those names free, so its closure matches that
   of [k] (FlambdaCompareProofs.cont_matches): applied to the same
   values the two come to states that are equivalent
   (FlambdaCompareProofs.apply_cont_value_match). The handler that is
   put in its place binds its parameters and applies [k] to them, which
   is the closure of [k] applied to the values given. So the two
   closures of the continuation whose handler is replaced apply alike,
   and one may stand for the other in any body
   (FlambdaContAlikeProofs.let_cont_handler_alike). Each layer of the
   term is a context around that
   (FlambdaContextProofs.plug_equivalent). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaCompareSpec FlambdaCompareProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaUnusedContsSpec FlambdaUnusedContsProofs FlambdaContAlikeProofs
  FlambdaSinkContsProofs FlambdaSameHandlersSpec.
Import ListNotations.

(* Distinct parameters, once bound to values, have those values. *)
Lemma eval_bound_params (params : list variable) :
  ∀ (vs : list value) (vars bound : env),
    bind_params params vs vars = Some bound ->
    distinct params = true ->
    eval_simples bound (map Var params) = Some vs.
Proof.
  induction params as [ | param params IH]; intros [ | v vs] vars bound;
    simpl; try discriminate.
  - intros _ _. reflexivity.
  - intros Hbind Hdistinct.
    apply andb_true_iff in Hdistinct as [Hfirst Hdistinct].
    apply negb_true_iff in Hfirst.
    rewrite (lookup_bind_params_other params vs _ bound param Hbind).
    + simpl. rewrite Pos.eqb_refl. simpl.
      rewrite (IH vs _ bound Hbind Hdistinct). reflexivity.
    + intros Hin. apply among_In in Hin. congruence.
Qed.

(* The closure of a handler that passes the test matches that of [k]. *)
Lemma same_handler_matches (h h' : continuation_handler)
    (xs : list variable) (ks : list continuation) (vars vars0 : env)
    (conts conts0 : cont_env) :
  same_handler h xs ks h' = true ->
  (∀ x : variable, x ∉ xs -> lookup x vars = lookup x vars0) ->
  (∀ c : continuation, c ∉ ks -> resolve c conts = resolve c conts0) ->
  cont_matches (Closure h' vars conts []) (Closure h vars0 conts0 []).
Proof.
  intros Hsame Hvars Hconts. unfold same_handler in Hsame.
  apply andb_true_iff in Hsame as [Hsame Hks].
  apply andb_true_iff in Hsame as [Hsame Hxs].
  apply andb_true_iff in Hsame as [Heq _].
  apply negb_true_iff in Hxs, Hks.
  apply (Closure_matches [] [] [] h' h vars vars0 conts conts0 [] []).
  - reflexivity.
  - exact Heq.
  - reflexivity.
  - intros y y' [Hy | []] Hnames. apply names_match_nil in Hnames.
    subst y'. apply Hvars. intros Hin.
    apply var_free_in_handler_b_sound in Hy.
    assert (Hsome : existsb (fun x => var_free_in_handler_b x h') xs = true)
      by (apply existsb_exists; exists y; split; assumption).
    congruence.
  - intros c c' [[Hc | []] _] Hnames. apply names_match_nil in Hnames.
    subst c'. rewrite (Hconts c); [ apply Same_matches | ].
    intros Hin. apply cont_free_in_handler_b_sound in Hc.
    assert (Hsome : existsb (fun c => cont_free_in_handler_b c h') ks = true)
      by (apply existsb_exists; exists c; split; assumption).
    congruence.
Qed.

(* A handler whose closure matches that of [k], and the handler that
   applies [k] to its parameters, apply alike where [k] is that
   closure. *)
Lemma jump_applies_alike (k : continuation) (h : continuation_handler)
    (params : list variable) (metadata : handler_metadata) (handler : expr)
    (vars vars0 : env) (conts conts0 : cont_env) :
  resolve k conts = Closure h vars0 conts0 [] ->
  distinct params = true ->
  cont_matches (Closure (Handler params metadata handler) vars conts [])
    (Closure h vars0 conts0 []) ->
  applies_alike (Closure (Handler params metadata handler) vars conts [])
    (Closure
       (Handler params metadata
          (Apply_cont (Apply_cont_expr k (map Var params) None)))
       vars conts []).
Proof.
  intros Hk Hdistinct Hmatch vs statics hp rd.
  pose proof (apply_cont_value_match _ _ vs statics statics hp rd Hmatch
    (statics_match_same statics)) as Hrelated.
  remember (apply_cont_value (Closure h vars0 conts0 []) vs statics hp rd)
    as result eqn:Happly.
  revert Hrelated. simpl.
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
    [ | intros _; exact I ].
  destruct (bind_params params vs vars) as [handler_vars | ] eqn:Hbind;
    [ | intros _; exact I ].
  destruct result as [s' | k0 vs0 hp0 rd0 | ]; simpl; try contradiction.
  intros Hrelated.
  apply (equivalent_trans _ s'); [ apply matched_equivalent, Hrelated | ].
  apply equivalent_sym, step_equivalent.
  rewrite step_apply_cont. cbn [ac_args ac_continuation].
  rewrite (eval_bound_params params vs vars handler_vars Hbind Hdistinct).
  unfold apply_continuation. rewrite Hk. symmetry. exact Happly.
Qed.

(* Under environments that are those of the binding of [k] but for the
   names bound since. *)
Fixpoint share_with_equivalent (e : expr) {struct e} :
  ∀ (k : continuation) (h : continuation_handler) (xs : list variable)
    (ks : list continuation) (vars vars0 : env) (conts conts0 : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    resolve k conts = Closure h vars0 conts0 [] ->
    (∀ x : variable, x ∉ xs -> lookup x vars = lookup x vars0) ->
    (∀ c : continuation, c ∉ ks -> resolve c conts = resolve c conts0) ->
    equivalent (State e vars conts statics hp rd)
      (State (share_with k h xs ks e) vars conts statics hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k' [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros k h xs ks vars vars0 conts conts0 statics hp rd Hk Hvars Hconts;
    cbn [share_with]; try apply equivalent_refl.
  - (* Let *)
    apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply (share_with_equivalent body k h (x :: xs) ks _ vars0 _ conts0);
      [ exact Hk | | exact Hconts ].
    intros y Hy. simpl.
    destruct (Pos.eqb_spec y x) as [-> | _];
      [ exfalso; apply Hy; left; reflexivity | ].
    apply Hvars. intros Hin. apply Hy. right. exact Hin.
  - (* Let_static *)
    apply let_static_equivalent_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | exact I ].
    apply (share_with_equivalent body k h xs ks _ vars0 _ conts0);
      assumption.
  - (* Let_cont_non_recursive *)
    destruct (Pos.eqb_spec k' k) as [-> | Hne]; [ apply equivalent_refl | ].
    match goal with
    | |- equivalent _ (State (Let_cont_non_recursive _ ?replaced _) _ _ _ _ _)
      =>
      apply (equivalent_trans _
        (State (Let_cont_non_recursive k' replaced body) vars conts statics
           hp rd))
    end.
    + destruct (same_handler h xs ks (Handler params metadata handler))
        eqn:Hsame; [ | apply equivalent_refl ].
      apply let_cont_handler_alike.
      apply (jump_applies_alike k h params metadata handler vars vars0 conts
        conts0 Hk).
      * unfold same_handler in Hsame.
        apply andb_true_iff in Hsame as [Hsame _].
        apply andb_true_iff in Hsame as [Hsame _].
        apply andb_true_iff in Hsame as [_ Hdistinct]. exact Hdistinct.
      * exact (same_handler_matches h _ xs ks vars vars0 conts conts0 Hsame
          Hvars Hconts).
    + apply let_cont_non_recursive_equivalent_step.
      apply (share_with_equivalent body k h xs (k' :: ks) _ vars0 _ conts0).
      * rewrite resolve_other; [ exact Hk | ].
        intros Heq. apply Hne. symmetry. exact Heq.
      * exact Hvars.
      * intros c Hc. rewrite resolve_other.
        -- apply Hconts. intros Hin. apply Hc. right. exact Hin.
        -- intros Heq. apply Hc. left. symmetry. exact Heq.
Qed.

(* Under the [Let_cont] that binds [k]. *)
Lemma share_under_equivalent (k : continuation) (h : continuation_handler)
    (body : expr) :
  expr_equivalent (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h (share_with k h [] [k] body)).
Proof.
  intros vars conts statics hp rd. apply let_cont_non_recursive_equivalent_step.
  apply (share_with_equivalent body k h [] [k] _ vars _ conts).
  - apply resolve_head.
  - intros x _. reflexivity.
  - intros c Hc. apply resolve_other. intros Heq. apply Hc. left.
    symmetry. exact Heq.
Qed.

Fixpoint share_handlers_equivalent_aux (e : expr) :
  expr_equivalent e (share_handlers e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [share_handlers]; cbv zeta;
    try (intros vars conts statics hp rd; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (share_handlers_equivalent_aux body)).
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (share_handlers_equivalent_aux body)).
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (share_handlers handler)) body)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body) _ _
        (not_many _) (share_handlers_equivalent_aux handler)).
    + apply (expr_equivalent_trans _
        (Let_cont_non_recursive k
           (Handler params metadata (share_handlers handler))
           (share_handlers body))).
      * exact (plug_equivalent
          (Let_cont_non_recursive_body k
             (Handler params metadata (share_handlers handler)) Hole)
          _ _ (not_many _) (share_handlers_equivalent_aux body)).
      * destruct (sharable (Handler params metadata (share_handlers handler)));
          [ apply share_under_equivalent
          | intros vars conts statics hp rd; apply equivalent_refl ].
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
      _ _ (not_many _) (share_handlers_equivalent_aux body)).
Qed.

Theorem share_handlers_equivalent : share_handlers_equivalent_statement.
Proof. exact share_handlers_equivalent_aux. Qed.
