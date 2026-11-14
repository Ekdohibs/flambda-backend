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

(* Proofs of the statements of FlambdaSinkContsSpec.v. Nothing here needs
   review.

   Each move of a [Let_cont] is an equivalence of two expressions on
   their own, and FlambdaContextProofs.plug_equivalent puts it where it
   is made.
   - Past a [Let], past a [Let_static] and past another [Let_cont] the
     two expressions step to the same body, under continuation
     environments whose closures hold the same handlers under
     environments that agree on the free names of those handlers
     (FlambdaNominalProofs.coincide_equivalent).
   - Into a handler the handler changes, so that lemma does not apply.
     The [Let_cont] moved is first copied to the head of the handler,
     the original being kept: the closure of the handler with the copy
     and the closure without apply alike, as the copy binds the name to
     what it already stood for there
     (FlambdaContAlikeProofs.let_cont_handler_alike). Then the original
     is not used, and goes (FlambdaNominalProofs.unused_let_cont). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaCompareSpec
  FlambdaCompareProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaContextSpec FlambdaContextProofs FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaUnusedContsSpec FlambdaUnusedContsProofs
  FlambdaContAlikeProofs FlambdaSinkContsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Closures of one handler under environments that agree                    *)
(* ======================================================================== *)

(* Two closures of one non-recursive handler match if their environments
   agree on the free names of the handler. *)
Lemma closures_match (h : continuation_handler) (vars vars' : env)
    (conts conts' : cont_env) :
  (∀ x : variable, var_free_in_handler x h -> lookup x vars = lookup x vars')
  ->
  (∀ k : continuation,
    cont_free_in_handler k h ->
    cont_matches (resolve k conts) (resolve k conts')) ->
  cont_matches (Closure h vars conts []) (Closure h vars' conts' []).
Proof.
  intros Hvars Hconts.
  apply (Closure_matches [] [] [] h h vars vars' conts conts' [] []).
  - reflexivity.
  - apply handler_eqb_refl; apply reflexive_map_nil.
  - reflexivity.
  - intros x x' [Hx | []] Hmatch. apply names_match_nil in Hmatch.
    subst x'. apply Hvars, Hx.
  - intros k k' [[Hk | []] _] Hmatch. apply names_match_nil in Hmatch.
    subst k'. apply Hconts, Hk.
Qed.

(* One expression under two continuation environments that differ in
   their first binding, of the same name to closures that match. *)
Lemma head_matches_equivalent (e : expr) (vars : env) (k : continuation)
    (c c' : cont_value) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  cont_matches c c' ->
  equivalent (State e vars ((k, c) :: conts) statics hp rd)
    (State e vars ((k, c') :: conts) statics hp rd).
Proof.
  intros Hc. apply coincide_equivalent; [ reflexivity | ].
  intros k' _. unfold conts_agree_at.
  change (rename_continuation identity_renaming k') with k'.
  destruct (Pos.eq_dec k' k) as [-> | Hne].
  - rewrite !resolve_head. exact Hc.
  - rewrite !resolve_other by exact Hne. apply Same_matches.
Qed.

(* A variable that is not a parameter is bound after the parameters as it
   was before. *)
Lemma lookup_bind_params_other (params : list variable) :
  ∀ (vs : list value) (vars bound : env) (x : variable),
    bind_params params vs vars = Some bound ->
    x ∉ params ->
    lookup x bound = lookup x vars.
Proof.
  induction params as [ | param params IH]; intros [ | v vs] vars bound x;
    simpl; try discriminate.
  - intros [= <-] _. reflexivity.
  - intros Hbind Hnotin. rewrite (IH vs _ _ x Hbind) by tauto. simpl.
    destruct (Pos.eqb_spec x param) as [-> | _]; [ tauto | reflexivity ].
Qed.

(* ======================================================================== *)
(* The moves                                                                *)
(* ======================================================================== *)

(* Past a [Let] whose variable is not free in the handler. *)
Lemma let_cont_past_let (k : continuation) (h : continuation_handler)
    (x : variable) (kind : flambda_kind) (d : named) (body : expr) :
  ~ var_free_in_handler x h ->
  expr_equivalent (Let_cont_non_recursive k h (Let x kind d body))
    (Let x kind d (Let_cont_non_recursive k h body)).
Proof.
  intros Hx vars conts statics hp rd.
  eapply equivalent_trans;
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
  apply equivalent_sym.
  eapply equivalent_trans;
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  apply head_matches_equivalent, closures_match.
  - intros y Hy. simpl.
    destruct (Pos.eqb_spec y x) as [-> | _]; [ contradiction | reflexivity ].
  - intros k' _. apply Same_matches.
Qed.

(* Past a [Let_static]: the two bind the same things in the other
   order. *)
Lemma let_cont_past_let_static (k : continuation) (h : continuation_handler)
    (bindings : list static_binding) (body : expr) :
  expr_equivalent (Let_cont_non_recursive k h (Let_static bindings body))
    (Let_static bindings (Let_cont_non_recursive k h body)).
Proof.
  intros vars conts statics hp rd.
  eapply equivalent_trans;
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  apply let_static_equivalent_step.
  destruct (eval_static_bindings vars bindings statics) as [extended | ];
    [ | exact I ].
  apply equivalent_sym, step_equivalent, step_let_cont_non_recursive.
Qed.

(* Past another [Let_cont], where neither handler uses the other's
   name. *)
Lemma let_cont_past_let_cont (k k' : continuation)
    (h h' : continuation_handler) (body : expr) :
  k <> k' ->
  ~ cont_free_in_handler k h' ->
  ~ cont_free_in_handler k' h ->
  expr_equivalent
    (Let_cont_non_recursive k h (Let_cont_non_recursive k' h' body))
    (Let_cont_non_recursive k' h' (Let_cont_non_recursive k h body)).
Proof.
  intros Hne Hk Hk' vars conts statics hp rd.
  eapply equivalent_trans;
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  eapply equivalent_trans;
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  apply equivalent_sym.
  eapply equivalent_trans;
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  eapply equivalent_trans;
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  apply coincide_equivalent; [ reflexivity | ].
  intros k'' _. unfold conts_agree_at.
  change (rename_continuation identity_renaming k'') with k''.
  destruct (Pos.eq_dec k'' k) as [-> | Hne''].
  - rewrite resolve_head, (resolve_other k k') by exact Hne.
    rewrite resolve_head. apply closures_match; [ reflexivity | ].
    intros k1 Hk1. rewrite resolve_other; [ apply Same_matches | ].
    intros ->. contradiction.
  - rewrite (resolve_other k'' k) by exact Hne''.
    destruct (Pos.eq_dec k'' k') as [-> | Hne'].
    + rewrite !resolve_head. apply closures_match; [ reflexivity | ].
      intros k1 Hk1. rewrite resolve_other; [ apply Same_matches | ].
      intros ->. contradiction.
    + rewrite !(resolve_other k'' k') by exact Hne'.
      rewrite (resolve_other k'' k) by exact Hne''. apply Same_matches.
Qed.

(* Into the handler of the next [Let_cont], which alone uses it. *)
Lemma let_cont_into_handler (k k' : continuation) (h : continuation_handler)
    (params' : list variable) (metadata' : handler_metadata)
    (handler' body : expr) :
  ~ cont_free_in_expr k body ->
  ~ cont_free_in_handler k h ->
  (∀ param : variable, param ∈ params' -> ~ var_free_in_handler param h) ->
  expr_equivalent
    (Let_cont_non_recursive k h
       (Let_cont_non_recursive k' (Handler params' metadata' handler') body))
    (Let_cont_non_recursive k'
       (Handler params' metadata' (Let_cont_non_recursive k h handler'))
       body).
Proof.
  intros Hbody Hh Hparams.
  apply (expr_equivalent_trans _
    (Let_cont_non_recursive k h
       (Let_cont_non_recursive k'
          (Handler params' metadata' (Let_cont_non_recursive k h handler'))
          body))).
  - (* the copy *)
    intros vars conts statics hp rd. apply
      let_cont_non_recursive_equivalent_step.
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
    + intros x Hx.
      apply (lookup_bind_params_other params' vs vars handler_vars x Hbind).
      intros Hin. exact (Hparams x Hin Hx).
    + intros k1 Hk1. rewrite resolve_other; [ apply Same_matches | ].
      intros ->. contradiction.
  - (* the original is not used any more *)
    apply unused_let_cont. simpl. tauto.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

Lemma params_not_free_sound (params : list variable)
    (h : continuation_handler) :
  params_not_free params h = true ->
  ∀ param : variable, param ∈ params -> ~ var_free_in_handler param h.
Proof.
  unfold params_not_free. rewrite forallb_forall.
  intros Hparams param Hin Hfree. specialize (Hparams param Hin).
  apply var_free_in_handler_b_sound in Hfree. rewrite Hfree in Hparams.
  discriminate.
Qed.

(* A [Let_cont] moved is equivalent to the [Let_cont] where it was. *)
Fixpoint sink_into_handler_equivalent (body : expr) {struct body} :
  ∀ (k : continuation) (h : continuation_handler) (sunk : expr),
    sink_into_handler k h body = Some sunk ->
    expr_equivalent (Let_cont_non_recursive k h body) sunk.
Proof.
  destruct body
    as [ x kind d body | bindings body | k' [params' metadata' handler'] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros k h sunk; simpl; try discriminate.
  - (* Let *)
    destruct (var_free_in_handler_b x h) eqn:Hfree; [ discriminate | ].
    destruct (sink_into_handler k h body) as [sunk' | ] eqn:Hsink;
      [ | discriminate ].
    intros [= <-].
    apply (expr_equivalent_trans _
      (Let x kind d (Let_cont_non_recursive k h body))).
    + apply let_cont_past_let. intros Hx.
      apply var_free_in_handler_b_sound in Hx. congruence.
    + exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
        (sink_into_handler_equivalent body k h sunk' Hsink)).
  - (* Let_static *)
    destruct (sink_into_handler k h body) as [sunk' | ] eqn:Hsink;
      [ | discriminate ].
    intros [= <-].
    apply (expr_equivalent_trans _
      (Let_static bindings (Let_cont_non_recursive k h body))).
    + apply let_cont_past_let_static.
    + exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
        (sink_into_handler_equivalent body k h sunk' Hsink)).
  - (* Let_cont_non_recursive *)
    destruct
      (negb (cont_free_b k body) && negb (cont_free_in_handler_b k h)
       && params_not_free params' h)
      eqn:Hinto.
    + (* into its handler *)
      apply andb_true_iff in Hinto as [Hinto Hparams].
      apply andb_true_iff in Hinto as [Hbody Hh].
      apply negb_true_iff in Hbody, Hh.
      assert (Hinto :
        expr_equivalent
          (Let_cont_non_recursive k h
             (Let_cont_non_recursive k' (Handler params' metadata' handler')
                body))
          (Let_cont_non_recursive k'
             (Handler params' metadata' (Let_cont_non_recursive k h handler'))
             body)).
      { apply let_cont_into_handler.
        - intros Hk. apply cont_free_b_sound in Hk. congruence.
        - intros Hk. apply cont_free_in_handler_b_sound in Hk. congruence.
        - apply params_not_free_sound, Hparams. }
      destruct (sink_into_handler k h handler') as [sunk' | ] eqn:Hsink.
      * intros [= <-]. apply (expr_equivalent_trans _ _ _ Hinto).
        exact (plug_equivalent
          (Let_cont_non_recursive_handler k' params' metadata' Hole body) _ _
          (not_many _)
          (sink_into_handler_equivalent handler' k h sunk' Hsink)).
      * destruct (applied_at_tail k h handler'); [ | discriminate ].
        intros [= <-]. exact Hinto.
    + (* past it *)
      destruct
        (negb (Pos.eqb k k') && negb (cont_free_b k handler')
         && negb (cont_free_in_handler_b k' h))
        eqn:Hpast; [ | discriminate ].
      destruct (sink_into_handler k h body) as [sunk' | ] eqn:Hsink;
        [ | discriminate ].
      intros [= <-].
      apply andb_true_iff in Hpast as [Hpast Hk'].
      apply andb_true_iff in Hpast as [Hne Hk].
      apply negb_true_iff in Hne, Hk, Hk'. apply Pos.eqb_neq in Hne.
      apply (expr_equivalent_trans _
        (Let_cont_non_recursive k' (Handler params' metadata' handler')
           (Let_cont_non_recursive k h body))).
      * apply let_cont_past_let_cont; [ exact Hne | | ].
        -- intros Hfree. simpl in Hfree.
           apply cont_free_b_sound in Hfree. congruence.
        -- intros Hfree. apply cont_free_in_handler_b_sound in Hfree.
           congruence.
      * exact (plug_equivalent
          (Let_cont_non_recursive_body k'
             (Handler params' metadata' handler') Hole)
          _ _ (not_many _)
          (sink_into_handler_equivalent body k h sunk' Hsink)).
Qed.

(* Each layer is a context around what the induction gives; then the
   [Let_cont] is moved. *)
Fixpoint sink_let_conts_equivalent_aux (e : expr) :
  expr_equivalent e (sink_let_conts e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try (intros vars conts statics hp rd; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (sink_let_conts_equivalent_aux body)).
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (sink_let_conts_equivalent_aux body)).
  - assert (Hlayers :
      expr_equivalent
        (Let_cont_non_recursive k (Handler params metadata handler) body)
        (Let_cont_non_recursive k
           (Handler params metadata (sink_let_conts handler))
           (sink_let_conts body))).
    { apply (expr_equivalent_trans _
        (Let_cont_non_recursive k
           (Handler params metadata (sink_let_conts handler)) body)).
      - exact (plug_equivalent
          (Let_cont_non_recursive_handler k params metadata Hole body) _ _
          (not_many _) (sink_let_conts_equivalent_aux handler)).
      - exact (plug_equivalent
          (Let_cont_non_recursive_body k
             (Handler params metadata (sink_let_conts handler)) Hole)
          _ _ (not_many _) (sink_let_conts_equivalent_aux body)). }
    destruct
      (sink_into_handler k (Handler params metadata (sink_let_conts handler))
         (sink_let_conts body))
      as [sunk | ] eqn:Hsink; [ | exact Hlayers ].
    exact (expr_equivalent_trans _ _ _ Hlayers
      (sink_into_handler_equivalent _ _ _ _ Hsink)).
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole) _ _
      (not_many _) (sink_let_conts_equivalent_aux body)).
Qed.

Theorem sink_let_conts_equivalent : sink_let_conts_equivalent_statement.
Proof. exact sink_let_conts_equivalent_aux. Qed.
