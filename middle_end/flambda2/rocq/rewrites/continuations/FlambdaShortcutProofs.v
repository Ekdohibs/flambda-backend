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

(* Proofs of the statements of FlambdaShortcutSpec.v. Nothing here needs
   review.

   The two [Switch]es take the same arm. Where the arm was changed, the
   first applies the continuation bound, whose handler then runs the
   [Apply_cont] under the variables and continuations of the
   [Let_cont]; the second runs that [Apply_cont] at once, under the same
   variables and with the continuation bound still in scope, which the
   [Apply_cont] does not apply. Each layer of the term is a context
   around that (FlambdaContextProofs.plug_equivalent). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaShortcutSpec.
Import ListNotations.

Lemma shortcut_arm_fst (k : continuation) (target : apply_cont) (i : Z)
    (ac : apply_cont) :
  fst (shortcut_arm k target (i, ac)) = i.
Proof.
  unfold shortcut_arm. simpl. destruct (ac_args ac); [ | reflexivity ].
  destruct (Pos.eqb (ac_continuation ac) k && no_trap_action ac);
    reflexivity.
Qed.

Lemma find_arm_shortcut (k : continuation) (target : apply_cont) (i : Z)
    (arms : list (Z * apply_cont)) :
  find_arm i (map (shortcut_arm k target) arms)
  = option_map (fun ac => snd (shortcut_arm k target (i, ac)))
      (find_arm i arms).
Proof.
  induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
  cbn [map find_arm].
  pose proof (shortcut_arm_fst k target d ac) as Hfst.
  destruct (shortcut_arm k target (d, ac)) as [d' ac'] eqn:Harm.
  simpl in Hfst. subst d'.
  destruct (Z.eqb_spec i d) as [-> | _]; [ | exact IH ].
  cbn [option_map]. rewrite Harm. reflexivity.
Qed.

(* The two [Switch]es under the binding of the continuation. *)
Lemma shortcut_arms_equivalent (k : continuation)
    (metadata : handler_metadata) (target : apply_cont) (s : simple)
    (arms : list (Z * apply_cont)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  hm_params_kinds metadata = [] ->
  ac_continuation target <> k ->
  equivalent
    (State (Switch s arms) vars
       ((k, Closure (Handler [] metadata (Apply_cont target)) vars conts [])
        :: conts)
       statics hp rd)
    (State (Switch s (map (shortcut_arm k target) arms)) vars
       ((k, Closure (Handler [] metadata (Apply_cont target)) vars conts [])
        :: conts)
       statics hp rd).
Proof.
  intros Hno_kinds Hother.
  destruct (eval_simple vars s) as [v | ] eqn:Hs;
    [ | apply step_eq_equivalent; rewrite !step_switch, Hs; reflexivity ].
  destruct (naked_immediate_of_value v) as [i | ] eqn:Hi;
    [ | apply step_eq_equivalent; rewrite !step_switch, Hs, Hi;
        reflexivity ].
  destruct (find_arm i arms) as [ac | ] eqn:Hfind;
    [ | apply step_eq_equivalent;
        rewrite !step_switch, Hs, Hi, find_arm_shortcut, Hfind; reflexivity ].
  eapply equivalent_trans;
    [ apply step_equivalent; rewrite step_switch, Hs, Hi, Hfind;
      reflexivity
    | ].
  eapply equivalent_trans;
    [ | apply equivalent_sym, step_equivalent;
        rewrite step_switch, Hs, Hi, find_arm_shortcut, Hfind; reflexivity ].
  unfold shortcut_arm. destruct ac as [applied args trap]. simpl.
  destruct args; [ | apply equivalent_refl ].
  destruct (Pos.eqb_spec applied k) as [-> | _]; [ | apply equivalent_refl ].
  simpl. destruct (no_trap_action _); [ | apply equivalent_refl ]. simpl.
  apply
    (equivalent_trans _ (State (Apply_cont target) vars conts statics hp rd)).
  - apply step_equivalent.
    apply
      (step_apply_cont_handler _ _ _ _ _ rd_ [] [] metadata (Apply_cont target)
      vars conts []);
      [ reflexivity | apply resolve_head
      | rewrite Hno_kinds; reflexivity | reflexivity ].
  - apply equivalent_sym, step_eq_equivalent, apply_cont_skip, Hother.
Qed.

Lemma shortcut_switch_equivalent (k : continuation) (h : continuation_handler)
    (body : expr) :
  expr_equivalent (Let_cont_non_recursive k h body) (shortcut_switch k h body).
Proof.
  unfold shortcut_switch.
  destruct h as [[ | param params] metadata handler];
    try (intros vars conts statics hp rd; apply equivalent_refl).
  destruct handler as [ | | | | | target | | ];
    try (intros vars conts statics hp rd; apply equivalent_refl).
  destruct body; try (intros vars conts statics hp rd; apply equivalent_refl).
  destruct (hm_params_kinds metadata) eqn:Hno_kinds;
    [ | intros vars conts statics hp rd; apply equivalent_refl ].
  destruct (Pos.eqb_spec (ac_continuation target) k) as [ | Hother];
    [ intros vars conts statics hp rd; apply equivalent_refl | ].
  cbn [orb].
  destruct (hm_is_exn_handler metadata);
    [ intros vars conts statics hp rd; apply equivalent_refl | ].
  cbn [orb].
  destruct (no_trap_action target);
    [ | intros vars conts statics hp rd; apply equivalent_refl ].
  cbn [negb]. intros vars conts statics hp rd.
  destruct (arms_apply k (map (shortcut_arm k target) arms)) eqn:Happly.
  - apply let_cont_non_recursive_equivalent_step, shortcut_arms_equivalent;
      [ exact Hno_kinds | exact Hother ].
  - eapply equivalent_trans;
      [ apply step_equivalent, step_let_cont_non_recursive | ].
    eapply equivalent_trans;
      [ apply shortcut_arms_equivalent; [ exact Hno_kinds | exact Hother ]
      | ].
    apply switch_skip, Happly.
Qed.

Fixpoint shortcut_switches_equivalent_aux (e : expr) :
  expr_equivalent e (shortcut_switches e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try (intros vars conts statics hp rd; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (shortcut_switches_equivalent_aux body)).
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (shortcut_switches_equivalent_aux body)).
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (shortcut_switches handler)) body)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body) _ _
        (not_many _) (shortcut_switches_equivalent_aux handler)).
    + apply (expr_equivalent_trans _
        (Let_cont_non_recursive k
           (Handler params metadata (shortcut_switches handler))
           (shortcut_switches body))).
      * exact (plug_equivalent
          (Let_cont_non_recursive_body k
             (Handler params metadata (shortcut_switches handler)) Hole)
          _ _ (not_many _) (shortcut_switches_equivalent_aux body)).
      * apply shortcut_switch_equivalent.
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
      _ _ (not_many _) (shortcut_switches_equivalent_aux body)).
Qed.

Theorem shortcut_switches_equivalent : shortcut_switches_equivalent_statement.
Proof. exact shortcut_switches_equivalent_aux. Qed.
