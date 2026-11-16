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

(* Proof of the statement of FlambdaLiftContsScopeSpec.v. Nothing here
   needs review.

   The steps are those of FlambdaLiftContsProofs.v: a continuation
   takes the parameters that it reads, where it is, and then goes out
   of the handler; each step keeps scope. *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaBindersSpec
  FlambdaNormaliseSpec FlambdaUnusedContsSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs FlambdaLiftContsSpec
  FlambdaLiftContsProofs FlambdaLiftContsScopeSpec.
Import ListNotations.

(* ======================================================================== *)
(* The applications given more arguments                                    *)
(* ======================================================================== *)

Lemma give_to_free_variables (j : continuation) (ps : list variable)
    (ac : apply_cont) :
  incl (free_variables_of_apply_cont (give_to j ps ac))
    (free_variables_of_apply_cont ac ++ ps).
Proof.
  unfold give_to.
  destruct (Pos.eqb (ac_continuation ac) j); [ | apply incl_appl, incl_refl ].
  unfold free_variables_of_apply_cont, free_variables_of_simples.
  cbn [ac_args]. rewrite flat_map_app. intros a Ha.
  apply in_or_app.
  apply in_app_or in Ha as [Ha | Ha]; [ right | left; exact Ha ].
  apply in_flat_map in Ha as (s & Hs & Ha).
  apply in_map_iff in Hs as (p & <- & Hp). destruct Ha as [<- | []]. exact Hp.
Qed.

Lemma give_to_continuations (j : continuation) (ps : list variable)
    (ac : apply_cont) :
  apply_cont_continuations (give_to j ps ac) = apply_cont_continuations ac.
Proof.
  unfold give_to. destruct (Pos.eqb (ac_continuation ac) j); reflexivity.
Qed.

(* The code of a term is closed as it was; its free variables are those
   that it had and the variables given; its free continuations are
   those that it had. *)
Fixpoint give_scope (j : continuation) (ps : list variable) (e : expr)
    {struct e} :
  code_closed (give j ps e) = code_closed e
  ∧ incl (free_variables_of_expr (give j ps e))
      (free_variables_of_expr e ++ ps)
  ∧ free_continuations_of_expr (give j ps e) = free_continuations_of_expr e
with give_in_handler_scope (j : continuation) (ps : list variable)
    (h : continuation_handler) {struct h} :
  handler_code_closed (give_in_handler j ps h) = handler_code_closed h
  ∧ incl (free_variables_of_handler (give_in_handler j ps h))
      (free_variables_of_handler h ++ ps)
  ∧ free_continuations_of_handler (give_in_handler j ps h)
    = free_continuations_of_handler h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [give code_closed free_variables_of_expr
           free_continuations_of_expr].
    + destruct (give_scope j ps body) as (Hc & Hv & Hk).
      split; [ exact Hc | split; [ | exact Hk ] ].
      apply incl_app_under; [ apply incl_appl, incl_refl | ].
      intros a Ha. apply in_without in Ha as [Ha Hy].
      apply Hv, in_app_or in Ha. apply in_or_app.
      destruct Ha as [Ha | Ha]; [ left | right; exact Ha ].
      apply in_without. split; assumption.
    + destruct (give_scope j ps body) as (Hc & Hv & Hk).
      rewrite Hc, Hk. split; [ reflexivity | split; [ | reflexivity ] ].
      apply incl_app_under; [ apply incl_appl, incl_refl | exact Hv ].
    + destruct (give_in_handler_scope j ps hh) as (Hc & Hv & Hk).
      destruct (give_scope j ps body) as (Hc' & Hv' & Hk').
      rewrite Hc, Hc', Hk, Hk'.
      split; [ reflexivity | split; [ | reflexivity ] ].
      apply incl_app_under; assumption.
    + destruct (give_scope j ps body) as (Hc & Hv & Hk).
      assert (Hhandlers :
        forallb (fun '(_, h) => handler_code_closed h)
          (map (fun '(k, h) => (k, give_in_handler j ps h)) handlers)
        = forallb (fun '(_, h) => handler_code_closed h) handlers
        ∧ incl
            (flat_map (fun '(_, h) => free_variables_of_handler h)
               (map (fun '(k, h) => (k, give_in_handler j ps h)) handlers))
            (flat_map (fun '(_, h) => free_variables_of_handler h) handlers
             ++ ps)
        ∧ flat_map (fun '(_, h) => free_continuations_of_handler h)
            (map (fun '(k, h) => (k, give_in_handler j ps h)) handlers)
          = flat_map (fun '(_, h) => free_continuations_of_handler h)
              handlers
        ∧ map fst (map (fun '(k, h) => (k, give_in_handler j ps h)) handlers)
          = map fst handlers).
      { induction handlers as [ | [k' hh] handlers IH];
          cbn [map forallb flat_map fst].
        - split; [ reflexivity | ].
          split; [ intros a [] | split; reflexivity ].
        - destruct (give_in_handler_scope j ps hh) as (Hc1 & Hv1 & Hk1).
          destruct IH as (Hc2 & Hv2 & Hk2 & Hfst).
          rewrite Hc1, Hc2, Hk1, Hk2, Hfst.
          split; [ reflexivity | ].
          split; [ apply incl_app_under; assumption | split; reflexivity ]. }
      destruct Hhandlers as (Hc' & Hv' & Hk' & Hfst).
      rewrite Hc, Hc', Hk, Hk', Hfst.
      split; [ reflexivity | split; [ | reflexivity ] ].
      apply incl_app_under; assumption.
    + split; [ reflexivity | split; [ apply incl_appl, incl_refl | ] ].
      reflexivity.
    + split; [ reflexivity | ].
      split; [ apply give_to_free_variables | apply give_to_continuations ].
    + split; [ reflexivity | ].
      split.
      * apply incl_app_under; [ apply incl_appl, incl_refl | ].
        induction arms as [ | [n ac] arms IH]; [ intros a [] | ].
        cbn [map flat_map snd].
        apply incl_app_under; [ apply give_to_free_variables | exact IH ].
      * induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
        cbn [map flat_map snd]. rewrite give_to_continuations, IH.
        reflexivity.
    + split; [ reflexivity | split; [ intros a [] | reflexivity ] ].
  - destruct h as [params metadata handler].
    cbn [give_in_handler handler_code_closed free_variables_of_handler
         free_continuations_of_handler].
    destruct (give_scope j ps handler) as (Hc & Hv & Hk).
    split; [ exact Hc | split; [ | exact Hk ] ].
    intros a Ha. apply in_without in Ha as [Ha Hparams].
    apply Hv, in_app_or in Ha. apply in_or_app.
    destruct Ha as [Ha | Ha]; [ left | right; exact Ha ].
    apply in_without. split; assumption.
Qed.

(* ======================================================================== *)
(* One continuation out of a handler                                        *)
(* ======================================================================== *)

Lemma read_params_among (e : expr) (params : list variable) :
  ∀ (kinds : list flambda_kind) (p : variable),
    In p (map fst (read_params params kinds e)) -> In p params.
Proof.
  intros kinds p Hin. apply among_In. apply among_In in Hin.
  exact (among_read_params p e params kinds Hin).
Qed.

(* It takes the parameters that it reads, where it is. *)
Lemma take_params_scoped (k j : continuation) (params q ps : list variable)
    (metadata mdj mdj' : handler_metadata) (h rest body : expr) :
  incl ps params ->
  scoped
    (Let_cont_non_recursive k
       (Handler params metadata
          (Let_cont_non_recursive j (Handler q mdj h) rest))
       body)
    (Let_cont_non_recursive k
       (Handler params metadata
          (Let_cont_non_recursive j (Handler (ps ++ q) mdj' h)
             (give j ps rest)))
       body).
Proof.
  intros Hps Hclosed.
  destruct (give_scope j ps rest) as (Hc & Hv & Hk).
  scope_in Hclosed. scope. rewrite Hc, Hk.
  split; [ exact Hclosed | split; [ | apply incl_refl ] ].
  apply incl_app_app; [ | apply incl_refl ].
  intros a Ha. apply in_without in Ha as [Ha Hparams]. apply in_without.
  split; [ | exact Hparams ].
  apply in_or_app. apply in_app_or in Ha as [Ha | Ha].
  - left. apply in_without in Ha as [Ha Hq]. apply in_without.
    split; [ exact Ha | ]. intros Hin. apply Hq, in_or_app. right. exact Hin.
  - apply Hv, in_app_or in Ha as [Ha | Ha]; [ right; exact Ha | ].
    destruct (Hparams (Hps a Ha)).
Qed.

(* It then reads no parameter of the handler around it, and goes out. *)
Lemma move_out_scoped (k j : continuation) (params : list variable)
    (metadata : handler_metadata) (hj : continuation_handler)
    (rest body : expr) :
  (∀ a : variable,
    In a (free_variables_of_handler hj) -> ~ In a params) ->
  scoped
    (Let_cont_non_recursive k
       (Handler params metadata (Let_cont_non_recursive j hj rest)) body)
    (Let_cont_non_recursive j hj
       (Let_cont_non_recursive k (Handler params metadata rest) body)).
Proof.
  intros Hfree Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed as [Hhandler Hbody].
  apply andb_true_iff in Hhandler as [Hhj Hrest].
  rewrite Hhj, Hrest, Hbody. split; [ reflexivity | split ].
  - intros a Ha. apply in_or_app.
    apply in_app_or in Ha as [Ha | Ha].
    + left. apply in_without. split; [ | exact (Hfree a Ha) ].
      apply in_or_app. left. exact Ha.
    + apply in_app_or in Ha as [Ha | Ha]; [ left | right; exact Ha ].
      apply in_without in Ha as [Ha Hparams]. apply in_without.
      split; [ | exact Hparams ]. apply in_or_app. right. exact Ha.
  - intros a Ha. apply in_or_app.
    apply in_app_or in Ha as [Ha | Ha].
    + left. apply in_or_app. left. exact Ha.
    + apply in_without in Ha as [Ha Hj].
      apply in_app_or in Ha as [Ha | Ha]; [ left | right; exact Ha ].
      apply in_or_app. right. apply in_without. split; assumption.
Qed.

(* ======================================================================== *)
(* The continuations of a handler in turn                                   *)
(* ======================================================================== *)

Lemma lift_spine_scoped (k : continuation) (params : list variable)
    (metadata : handler_metadata) (body : expr) :
  length (hm_params_kinds metadata) = length params ->
  ∀ (handler : expr) (moved : list (continuation * list variable))
    (lifted : list (continuation * continuation_handler)) (switch : expr),
    lift_spine body params (hm_params_kinds metadata) moved handler
    = Some (lifted, switch) ->
    scoped
      (Let_cont_non_recursive k
         (Handler params metadata (give_all moved handler)) body)
      (bind_lifted lifted
         (Let_cont_non_recursive k (Handler params metadata switch) body)).
Proof.
  intros Hlength handler.
  induction handler
    as [ y kind d rest _ | bindings rest _ | j [q mdj h] rest IH
       | handlers rest _ | ap | ac | scrutinee arms | ];
    intros moved lifted switch; cbn [lift_spine]; try discriminate.
  - (* a continuation of the handler *)
    set (h' := give_all moved h).
    set (read := read_params params (hm_params_kinds metadata) h').
    destruct
      (negb (hm_is_exn_handler mdj) && distinct (params ++ q)
       && negb (cont_free_b j body) && negb (cont_free_b j h')
       && only_applied j (map fst read) true (give_all moved rest));
      [ | discriminate ].
    destruct
      (lift_spine body params (hm_params_kinds metadata)
         ((j, map fst read) :: moved) rest)
      as [[others switch'] | ] eqn:Hrest; [ | discriminate ].
    intros [= <- <-]. cbn [bind_lifted fold_right].
    rewrite give_all_let_cont. fold h'.
    eapply scoped_trans.
    { apply
        (take_params_scoped k j params q (map fst read) metadata mdj
           {| hm_params_kinds := map snd read ++ hm_params_kinds mdj;
              hm_is_exn_handler := false |}
           h' (give_all moved rest) body).
      intros p Hp. exact (read_params_among h' params _ p Hp). }
    intros Hclosed.
    assert (Hh' : code_closed h' = true).
    { scope_in Hclosed. apply andb_true_iff in Hclosed as [Hclosed _].
      apply andb_true_iff in Hclosed as [Hclosed _]. exact Hclosed. }
    revert Hclosed.
    eapply scoped_trans.
    { apply move_out_scoped. intros a Ha Hin.
      cbn [free_variables_of_handler] in Ha.
      apply in_without in Ha as [Ha Hnot].
      apply Hnot, in_or_app. left.
      apply in_read_params; [ exact Hlength | exact Hin | ].
      exact (var_free_b_free a h' Hh' Ha). }
    apply scoped_let_cont; [ apply scoped_handler_refl | ].
    exact (IH ((j, map fst read) :: moved) others switch' Hrest).
  - (* the [Switch] *)
    destruct scrutinee as [p | sym | c]; try discriminate.
    destruct (among p params); [ | discriminate ].
    intros [= <- <-]. apply scoped_refl.
Qed.

Lemma lift_conts_scoped (k : continuation) (h : continuation_handler)
    (body lifted : expr) :
  lift_conts k h body = Some lifted ->
  scoped (Let_cont_non_recursive k h body) lifted.
Proof.
  destruct h as [params metadata handler]. unfold lift_conts.
  destruct (tested params handler) as [i | ]; [ | discriminate ].
  destruct
    (negb (hm_is_exn_handler metadata) && distinct params
     && Nat.eqb (length (hm_params_kinds metadata)) (length params)
     && all_give_immediate k i body
     && negb (number_of_a_scrutinee k i body))
    eqn:Htests; [ | discriminate ].
  apply andb_true_iff in Htests as [Htests _].
  apply andb_true_iff in Htests as [Htests _].
  apply andb_true_iff in Htests as [_ Hlength].
  apply Nat.eqb_eq in Hlength.
  destruct (lift_spine body params (hm_params_kinds metadata) [] handler)
    as [[[ | first others] switch] | ] eqn:Hspine; try discriminate.
  destruct (read_where_given k i params switch body); [ | discriminate ].
  intros [= <-].
  exact (lift_spine_scoped k params metadata body Hlength handler []
    (first :: others) switch Hspine).
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

Fixpoint lift_handler_conts_scoped (e : expr) :
  scoped e (lift_handler_conts e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [lift_handler_conts]; try apply scoped_refl.
  - apply scoped_let; [ apply incl_refl | apply lift_handler_conts_scoped ].
  - apply scoped_let_static;
      [ apply scoped_bindings_refl | apply lift_handler_conts_scoped ].
  - assert (Hlayers :
      scoped
        (Let_cont_non_recursive k (Handler params metadata handler) body)
        (Let_cont_non_recursive k
           (Handler params metadata (lift_handler_conts handler))
           (lift_handler_conts body))).
    { apply scoped_let_cont;
        [ apply scoped_handler_intro | ]; apply lift_handler_conts_scoped. }
    destruct
      (lift_conts k (Handler params metadata (lift_handler_conts handler))
         (lift_handler_conts body))
      as [lifted | ] eqn:Hlift; [ | exact Hlayers ].
    exact (scoped_trans _ _ _ Hlayers (lift_conts_scoped k _ _ lifted Hlift)).
  - apply scoped_let_cont_recursive;
      [ reflexivity | apply scoped_handlers_refl
      | apply lift_handler_conts_scoped ].
Qed.

Theorem lift_handler_conts_keeps_scope :
  lift_handler_conts_keeps_scope_statement.
Proof. apply keeps_scope_intro, lift_handler_conts_scoped. Qed.
