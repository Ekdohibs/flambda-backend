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

(* FlambdaBindConstParamsSpec.bind_const_params commutes with renaming
   and keeps scope: the lemmas from which
   FlambdaValidatorEquivarianceProofs.v and FlambdaStaticsScopeProofs.v
   prove the two statements. There is no statement here, and nothing
   here needs review.

   The pass compares names by equality only and makes none, and a
   renaming moves no symbol and no constant, so each of its functions
   gives, on a renamed term, the renamed result. For the scope: an
   application that loses an argument loses free variables and no
   continuation; the [Let] that the handler gains mentions no variable;
   and the parameter that the handler loses is bound by that [Let]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaUnaliasParamsSpec FlambdaUnaliasParamsProofs
  FlambdaUnaliasParamsNamesProofs FlambdaBindConstParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

(* A renaming is a permutation: two simples are the same once renamed
   if and only if they are. *)
Lemma same_simple_rename (r : renaming) (s s' : simple) :
  same_simple (rename_simple r s) (rename_simple r s') = same_simple s s'.
Proof.
  unfold same_simple. destruct (simple_eq_dec s s') as [-> | Hne].
  - destruct (simple_eq_dec (rename_simple r s') (rename_simple r s'));
      [ reflexivity | contradiction ].
  - destruct (simple_eq_dec (rename_simple r s) (rename_simple r s'))
      as [Heq | ]; [ | reflexivity ].
    exfalso. apply Hne.
    destruct s as [x | sym | c], s' as [x' | sym' | c']; simpl in Heq;
      try discriminate; try exact Heq.
    injection Heq as Heq. f_equal.
    exact (permute_injective (variable_permutation r) x x' Heq).
Qed.

Lemma is_plain_rename (r : renaming) (s : simple) :
  is_plain (rename_simple r s) = is_plain s.
Proof. destruct s; reflexivity. Qed.

Lemma apply_cont_gives_equivariant (r : renaming) (k : continuation)
    (i : nat) (s : simple) (ac : apply_cont) :
  apply_cont_gives (rename_continuation r k) i (rename_simple r s)
    (rename_apply_cont r ac)
  = apply_cont_gives k i s ac.
Proof.
  unfold apply_cont_gives, rename_apply_cont.
  cbn [ac_trap_action ac_continuation ac_args].
  rewrite rename_continuation_eqb, nth_error_map.
  f_equal.
  - destruct (ac_trap_action ac) as [action | ]; [ | reflexivity ].
    cbn [option_map].
    rewrite trap_action_handler_renamed, rename_continuation_eqb.
    reflexivity.
  - destruct (Pos.eqb (ac_continuation ac) k); [ | reflexivity ].
    destruct (nth_error (ac_args ac) i) as [s' | ]; cbn [option_map];
      [ | reflexivity ].
    apply same_simple_rename.
Qed.

Fixpoint all_give_equivariant (r : renaming) (k : continuation) (i : nat)
    (s : simple) (e : expr) {struct e} :
  all_give (rename_continuation r k) i (rename_simple r s) (rename_expr r e)
  = all_give k i s e
with handler_all_give_equivariant (r : renaming) (k : continuation)
    (i : nat) (s : simple) (h : continuation_handler) {struct h} :
  handler_all_give (rename_continuation r k) i (rename_simple r s)
    (rename_handler r h)
  = handler_all_give k i s h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr all_give]; try reflexivity.
    + apply all_give_equivariant.
    + apply all_give_equivariant.
    + rewrite rename_continuation_eqb, handler_all_give_equivariant,
        all_give_equivariant.
      reflexivity.
    + rewrite all_give_equivariant. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map forallb].
      rewrite rename_continuation_eqb, handler_all_give_equivariant, IH.
      reflexivity.
    + cbn [rename_apply ap_continuation ap_exn_continuation].
      rewrite !rename_continuation_eqb. reflexivity.
    + apply apply_cont_gives_equivariant.
    + induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map forallb fst snd].
      rewrite apply_cont_gives_equivariant, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_all_give]. apply all_give_equivariant.
Qed.

Definition rename_given_simple (r : renaming)
    (given : option (option simple)) : option (option simple) :=
  option_map (option_map (rename_simple r)) given.

Lemma given_by_equivariant (r : renaming) (k : continuation) (i : nat)
    (ac : apply_cont) :
  given_by (rename_continuation r k) i (rename_apply_cont r ac)
  = rename_given_simple r (given_by k i ac).
Proof.
  unfold given_by, rename_apply_cont.
  cbn [ac_continuation ac_args].
  rewrite rename_continuation_eqb, nth_error_map.
  destruct (Pos.eqb (ac_continuation ac) k); reflexivity.
Qed.

Fixpoint first_given_equivariant (r : renaming) (k : continuation)
    (i : nat) (e : expr) {struct e} :
  first_given (rename_continuation r k) i (rename_expr r e)
  = rename_given_simple r (first_given k i e)
with handler_first_given_equivariant (r : renaming) (k : continuation)
    (i : nat) (h : continuation_handler) {struct h} :
  handler_first_given (rename_continuation r k) i (rename_handler r h)
  = rename_given_simple r (handler_first_given k i h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr first_given]; try reflexivity.
    + apply first_given_equivariant.
    + apply first_given_equivariant.
    + rewrite handler_first_given_equivariant, first_given_equivariant.
      destruct (handler_first_given k i hh); reflexivity.
    + rewrite first_given_equivariant.
      assert (Hhandlers :
        first_answer
          (fun '(_, h) => handler_first_given (rename_continuation r k) i h)
          (map (fun '(k0, h) => (rename_continuation r k0, rename_handler r h))
             handlers)
        = rename_given_simple r
            (first_answer (fun '(_, h) => handler_first_given k i h)
               handlers)).
      { induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
        cbn [map first_answer]. rewrite handler_first_given_equivariant.
        destruct (handler_first_given k i hh); [ reflexivity | exact IH ]. }
      rewrite Hhandlers.
      destruct (first_answer (fun '(_, h) => handler_first_given k i h)
                  handlers);
        reflexivity.
    + apply given_by_equivariant.
    + induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map first_answer fst snd]. rewrite given_by_equivariant.
      destruct (given_by k i ac); [ reflexivity | exact IH ].
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_first_given].
    apply first_given_equivariant.
Qed.

Lemma bind_const_param_equivariant (r : renaming) (i : nat)
    (k : continuation) (h : continuation_handler) (body : expr) :
  bind_const_param i (rename_continuation r k) (rename_handler r h)
    (rename_expr r body)
  = rename_unaliased r (bind_const_param i k h body).
Proof.
  destruct h as [params metadata handler].
  cbn [rename_handler bind_const_param].
  rewrite nth_error_map, first_given_equivariant.
  destruct (nth_error params i) as [p | ]; [ | reflexivity ].
  cbn [option_map].
  destruct (nth_error (hm_params_kinds metadata) i) as [kind | ];
    [ | reflexivity ].
  destruct (first_given k i body) as [[s | ] | ]; try reflexivity.
  cbn [rename_given_simple option_map].
  rewrite is_plain_rename, all_give_equivariant, drop_nth_map,
    among_rename_variable, drop_arg_equivariant.
  destruct
    (is_plain s && negb (hm_is_exn_handler metadata) && all_give k i s body
     && negb (among p (drop_nth i params)));
    reflexivity.
Qed.

Lemma bind_const_from_equivariant (r : renaming) (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    bind_const_from i (rename_continuation r k) (rename_handler r h)
      (rename_expr r body)
    = (rename_handler r (fst (bind_const_from i k h body)),
       rename_expr r (snd (bind_const_from i k h body))).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [bind_const_from];
    [ reflexivity | ].
  rewrite bind_const_param_equivariant.
  destruct (bind_const_param i k h body) as [[h' body'] | ];
    cbn [rename_unaliased option_map]; apply IH.
Qed.

Fixpoint bind_const_params_equivariant_aux (r : renaming) (e : expr) :
  bind_const_params (rename_expr r e) = rename_expr r (bind_const_params e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler bind_const_params]; try reflexivity.
  - rewrite bind_const_params_equivariant_aux. reflexivity.
  - rewrite bind_const_params_equivariant_aux. reflexivity.
  - rewrite !bind_const_params_equivariant_aux, length_map.
    pose proof
      (bind_const_from_equivariant r (length params) k
         (Handler params metadata (bind_const_params handler))
         (bind_const_params body))
      as Hfrom.
    cbn [rename_handler] in Hfrom. rewrite Hfrom.
    destruct
      (bind_const_from (length params) k
         (Handler params metadata (bind_const_params handler))
         (bind_const_params body))
      as [h' body'].
    reflexivity.
  - rewrite bind_const_params_equivariant_aux. reflexivity.
Qed.

Lemma bind_const_params_renamed : rewrite_equivariant bind_const_params.
Proof. exact bind_const_params_equivariant_aux. Qed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

Lemma bind_const_param_scoped (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr) (h' : continuation_handler)
    (body' : expr) :
  bind_const_param i k h body = Some (h', body') ->
  scoped (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h' body').
Proof.
  destruct h as [params metadata handler]. unfold bind_const_param.
  destruct (nth_error params i) as [p | ] eqn:Hparam; [ | discriminate ].
  destruct (nth_error (hm_params_kinds metadata) i) as [kind | ];
    [ | discriminate ].
  destruct (first_given k i body) as [[s | ] | ]; try discriminate.
  destruct
    (is_plain s && negb (hm_is_exn_handler metadata) && all_give k i s body
     && negb (among p (drop_nth i params)))
    eqn:Hcond; [ | discriminate ].
  intros [= <- <-].
  apply andb_true_iff in Hcond as [Hcond _].
  apply andb_true_iff in Hcond as [Hcond _].
  apply andb_true_iff in Hcond as [Hplain _].
  assert (Hnone : free_variables_of_simple s = [])
    by (destruct s; [ discriminate | reflexivity | reflexivity ]).
  intros Hclosed.
  cbn [code_closed handler_code_closed] in Hclosed |- *.
  apply andb_true_iff in Hclosed as [Hhandler Hbody].
  destruct (drop_arg_scoped k i body Hbody) as (Hbody' & Hv & Hk).
  split; [ rewrite Hhandler, Hbody'; reflexivity | ].
  cbn [free_variables_of_expr free_variables_of_handler
       free_variables_of_named free_continuations_of_expr
       free_continuations_of_handler].
  rewrite Hnone. cbn [app].
  split.
  - intros a Ha. apply in_or_app. apply in_app_or in Ha as [Ha | Ha];
      [ | right; exact (Hv a Ha) ].
    apply in_without in Ha as [Ha Hnot].
    apply in_without in Ha as [Ha Hne].
    left. apply in_without. split; [ exact Ha | ].
    intros Hin.
    destruct (in_nth_or_drop_nth i params p a Hparam Hin) as [-> | Hin'];
      [ apply Hne; left; reflexivity | exact (Hnot Hin') ].
  - apply incl_app_app; [ apply incl_refl | apply incl_without, Hk ].
Qed.

Lemma bind_const_from_scoped (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    scoped (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k (fst (bind_const_from i k h body))
         (snd (bind_const_from i k h body))).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [bind_const_from];
    [ apply scoped_refl | ].
  destruct (bind_const_param i k h body) as [[h' body'] | ] eqn:Hparam;
    [ | apply IH ].
  exact (scoped_trans _ _ _
    (bind_const_param_scoped i k h body h' body' Hparam) (IH k h' body')).
Qed.

Fixpoint bind_const_params_scoped (e : expr) :
  scoped e (bind_const_params e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [bind_const_params]; try apply scoped_refl.
  - apply scoped_let; [ apply incl_refl | apply bind_const_params_scoped ].
  - apply scoped_let_static;
      [ apply scoped_bindings_refl | apply bind_const_params_scoped ].
  - pose proof
      (bind_const_from_scoped (length params) k
         (Handler params metadata (bind_const_params handler))
         (bind_const_params body))
      as Hfrom.
    destruct
      (bind_const_from (length params) k
         (Handler params metadata (bind_const_params handler))
         (bind_const_params body))
      as [h' body'].
    refine (scoped_trans _ _ _ _ Hfrom).
    apply scoped_let_cont;
      [ apply scoped_handler_intro, bind_const_params_scoped
      | apply bind_const_params_scoped ].
  - apply scoped_let_cont_recursive;
      [ reflexivity | apply scoped_handlers_refl
      | apply bind_const_params_scoped ].
Qed.

Lemma bind_const_params_scope : keeps_scope bind_const_params.
Proof. apply keeps_scope_intro, bind_const_params_scoped. Qed.
