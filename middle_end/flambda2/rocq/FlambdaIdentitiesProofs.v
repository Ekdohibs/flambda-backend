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



(* Proofs of the statements of FlambdaIdentitiesSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaIdentitiesSpec.
Import ListNotations.
Open Scope Z_scope.

(* ======================================================================== *)
(* The identities, on integers                                              *)
(* ======================================================================== *)

Lemma zero_in_range (width : Z) : 1 <= width -> in_range width 0 = true.
Proof.
  intros Hwidth. unfold in_range.
  assert (Hhalf : 0 < 2 ^ (width - 1)) by (apply Z.pow_pos_nonneg; lia).
  apply andb_true_iff. split; [ apply Z.leb_le | apply Z.ltb_lt ]; lia.
Qed.

Lemma minus_one_in_range (width : Z) :
  1 <= width -> in_range width (-1) = true.
Proof.
  intros Hwidth. unfold in_range.
  assert (Hhalf : 0 < 2 ^ (width - 1)) by (apply Z.pow_pos_nonneg; lia).
  apply andb_true_iff. split; [ apply Z.leb_le | apply Z.ltb_lt ]; lia.
Qed.

Lemma wrap_zero (width : Z) : 1 <= width -> wrap width 0 = 0.
Proof.
  intros Hwidth. apply wrap_of_in_range, zero_in_range; exact Hwidth.
Qed.

Lemma wrap_minus_one (width : Z) : 1 <= width -> wrap width (-1) = -1.
Proof.
  intros Hwidth. apply wrap_of_in_range, minus_one_in_range; exact Hwidth.
Qed.

Lemma quot_minus_one (a : Z) : Z.quot a (-1) = - a.
Proof.
  change (-1) with (- (1)). rewrite Z.quot_opp_r by lia.
  rewrite Z.quot_1_r. reflexivity.
Qed.

Lemma rem_minus_one (a : Z) : Z.rem a (-1) = 0.
Proof.
  change (-1) with (- (1)). rewrite Z.rem_opp_r by lia. apply Z.rem_1_r.
Qed.

(* An outcome is right of an operation whose other argument is the
   integer [a] and whose result is [r]. *)
Definition outcome_holds (width : Z) (o : outcome) (a r : Z) : Prop :=
  match o with
  | Exactly i => r = i
  | The_other_side => r = a
  | Negation_of_the_other_side => r = wrap width (0 - a)
  | Bitwise_not_of_the_other_side => r = wrap width (Z.lnot a)
  | Cannot_simplify => True
  end.

Section Identities.

Variable width : Z.
Hypothesis Hwidth : 1 <= width.

(* The known integer on the right. *)
Lemma symmetric_op_sound_r (op : binary_int_arith_op) (a this r : Z) :
  in_range width a = true ->
  eval_int_arith width op a this = Some r ->
  outcome_holds width (symmetric_op_one_side_unknown op this) a r.
Proof.
  intros Ha.
  destruct op; simpl; try (intros _; exact I).
  - (* [Add] *)
    destruct (Z.eqb_spec this 0) as [-> | _]; [ | intros _; exact I ].
    intros [= <-]. simpl. rewrite Z.add_0_r.
    apply wrap_of_in_range; assumption.
  - (* [Mul] *)
    destruct (Z.eqb_spec this 0) as [-> | _].
    { intros [= <-]. simpl. rewrite Z.mul_0_r. apply wrap_zero, Hwidth. }
    destruct (Z.eqb_spec this 1) as [-> | _].
    { intros [= <-]. simpl. rewrite Z.mul_1_r.
      apply wrap_of_in_range; assumption. }
    destruct (Z.eqb_spec this (-1)) as [-> | _]; [ | intros _; exact I ].
    intros [= <-]. simpl. f_equal. lia.
  - (* [And] *)
    destruct (Z.eqb_spec this (-1)) as [-> | _].
    { intros [= <-]. simpl. rewrite Z.land_m1_r.
      apply wrap_of_in_range; assumption. }
    destruct (Z.eqb_spec this 0) as [-> | _]; [ | intros _; exact I ].
    intros [= <-]. simpl. rewrite Z.land_0_r. apply wrap_zero, Hwidth.
  - (* [Or] *)
    destruct (Z.eqb_spec this (-1)) as [-> | _].
    { intros [= <-]. simpl. rewrite Z.lor_m1_r.
      apply wrap_minus_one, Hwidth. }
    destruct (Z.eqb_spec this 0) as [-> | _]; [ | intros _; exact I ].
    intros [= <-]. simpl. rewrite Z.lor_0_r.
    apply wrap_of_in_range; assumption.
  - (* [Xor] *)
    destruct (Z.eqb_spec this 0) as [-> | _].
    { intros [= <-]. simpl. rewrite Z.lxor_0_r.
      apply wrap_of_in_range; assumption. }
    destruct (Z.eqb_spec this (-1)) as [-> | _]; [ | intros _; exact I ].
    intros [= <-]. simpl. rewrite Z.lxor_m1_r. reflexivity.
Qed.

(* The operations of [symmetric_op_one_side_unknown] are commutative. *)
Lemma symmetric_op_comm (op : binary_int_arith_op) (a b : Z) :
  symmetric_op_one_side_unknown op b <> Cannot_simplify ->
  eval_int_arith width op b a = eval_int_arith width op a b.
Proof.
  destruct op; simpl; try (intros H; exfalso; apply H; reflexivity);
    intros _; f_equal; f_equal.
  - apply Z.add_comm.
  - apply Z.mul_comm.
  - apply Z.land_comm.
  - apply Z.lor_comm.
  - apply Z.lxor_comm.
Qed.

(* The known integer on the left. *)
Lemma symmetric_op_sound_l (op : binary_int_arith_op) (this a r : Z) :
  in_range width a = true ->
  eval_int_arith width op this a = Some r ->
  outcome_holds width (symmetric_op_one_side_unknown op this) a r.
Proof.
  intros Ha Heval.
  destruct (symmetric_op_one_side_unknown op this) eqn:Houtcome;
    try exact I;
    rewrite symmetric_op_comm in Heval by (rewrite Houtcome; discriminate);
    pose proof (symmetric_op_sound_r op a this r Ha Heval) as Hsound;
    rewrite Houtcome in Hsound; exact Hsound.
Qed.

Lemma op_lhs_unknown_sound (op : binary_int_arith_op) (a rhs r : Z) :
  in_range width a = true ->
  eval_int_arith width op a rhs = Some r ->
  outcome_holds width (op_lhs_unknown op rhs) a r.
Proof.
  intros Ha.
  destruct op as [ | | | s | s | | | ];
    try exact (symmetric_op_sound_r _ a rhs r Ha);
    cbn [op_lhs_unknown eval_int_arith].
  - (* [Sub] *)
    destruct (Z.eqb_spec rhs 0) as [-> | _]; [ | intros _; exact I ].
    intros [= <-]. simpl. rewrite Z.sub_0_r.
    apply wrap_of_in_range; assumption.
  - (* [Div] *)
    destruct (Z.eqb_spec rhs 1) as [-> | _].
    { simpl. intros [= <-].
      destruct s; simpl.
      - rewrite Z.quot_1_r. apply wrap_of_in_range; assumption.
      - change (unsigned width 1) with 1. rewrite Z.quot_1_r.
        apply wrap_unsigned; assumption. }
    destruct (Z.eqb_spec rhs (-1)) as [-> | _]; [ | intros _; exact I ].
    destruct s; [ | intros _; exact I ].
    change (-1 =? 0) with false.
    cbn [reading andb is_signed outcome_holds].
    intros [= <-]. f_equal. rewrite quot_minus_one. lia.
  - (* [Mod] *)
    destruct (Z.eqb_spec rhs 1) as [-> | _].
    { simpl. intros [= <-].
      destruct s; simpl.
      - rewrite Z.rem_1_r. apply wrap_zero, Hwidth.
      - change (unsigned width 1) with 1. rewrite Z.rem_1_r.
        apply wrap_zero, Hwidth. }
    destruct (Z.eqb_spec rhs (-1)) as [-> | _]; [ | intros _; exact I ].
    destruct s; [ | intros _; exact I ].
    change (-1 =? 0) with false.
    cbn [reading andb is_signed outcome_holds].
    intros [= <-]. rewrite rem_minus_one. apply wrap_zero, Hwidth.
Qed.

Lemma op_rhs_unknown_sound (op : binary_int_arith_op) (lhs a r : Z) :
  in_range width a = true ->
  eval_int_arith width op lhs a = Some r ->
  outcome_holds width (op_rhs_unknown op lhs) a r.
Proof.
  intros Ha.
  destruct op as [ | | | s | s | | | ];
    try exact (symmetric_op_sound_l _ lhs a r Ha);
    cbn [op_rhs_unknown]; try (intros _; exact I).
  (* [Sub] *)
  destruct (Z.eqb_spec lhs 0) as [-> | _]; [ | intros _; exact I ].
  intros [= <-]. reflexivity.
Qed.

(* A shift by 0. *)
Lemma shift_lhs_unknown_sound (op : int_shift_op) (a rhs : Z) :
  in_range width a = true ->
  outcome_holds width (shift_lhs_unknown op rhs) a
    (eval_int_shift width op a rhs).
Proof.
  intros Ha. unfold shift_lhs_unknown.
  destruct (Z.eqb_spec rhs 0) as [-> | _]; [ | exact I ].
  destruct op; cbn [eval_int_shift outcome_holds].
  - rewrite Z.shiftl_0_r. apply wrap_of_in_range; assumption.
  - rewrite Z.shiftr_0_r. apply wrap_unsigned; assumption.
  - rewrite Z.shiftr_0_r. apply wrap_of_in_range; assumption.
Qed.

Lemma shiftr_minus_one (amount : Z) :
  0 <= amount -> Z.shiftr (-1) amount = -1.
Proof.
  intros Hamount. rewrite Z.shiftr_div_pow2 by exact Hamount.
  assert (Hpow : 0 < 2 ^ amount) by (apply Z.pow_pos_nonneg; lia).
  symmetry. apply (Z.div_unique_pos _ _ _ (2 ^ amount - 1)); lia.
Qed.

(* A shift of 0, or of -1 to the right keeping the sign. *)
Lemma shift_rhs_unknown_sound (op : int_shift_op) (lhs amount : Z) :
  0 <= amount ->
  outcome_holds width (shift_rhs_unknown op lhs) amount
    (eval_int_shift width op lhs amount).
Proof.
  intros Hamount.
  destruct op; cbn [shift_rhs_unknown].
  - destruct (Z.eqb_spec lhs 0) as [-> | _]; [ | exact I ].
    cbn [eval_int_shift outcome_holds]. rewrite Z.shiftl_0_l.
    apply wrap_zero, Hwidth.
  - destruct (Z.eqb_spec lhs 0) as [-> | _]; [ | exact I ].
    cbn [eval_int_shift outcome_holds].
    change (unsigned width 0) with 0. rewrite Z.shiftr_0_l.
    apply wrap_zero, Hwidth.
  - destruct (Z.eqb_spec lhs 0) as [-> | _].
    { cbn [eval_int_shift outcome_holds]. rewrite Z.shiftr_0_l.
      apply wrap_zero, Hwidth. }
    destruct (Z.eqb_spec lhs (-1)) as [-> | _]; [ | exact I ].
    cbn [eval_int_shift outcome_holds].
    rewrite shiftr_minus_one by exact Hamount.
    apply wrap_minus_one, Hwidth.
Qed.

End Identities.

(* ======================================================================== *)
(* Applying one                                                             *)
(* ======================================================================== *)

Lemma int_of_value_const_of_int (kind : standard_int) (i : Z) :
  in_range (int_width kind) i = true ->
  int_of_value kind (Value_const (const_of_int kind i)) = Some i.
Proof.
  intros Hrange. destruct kind; simpl in *; rewrite Hrange; reflexivity.
Qed.

(* What an outcome that is right makes of the operation has the value of
   the operation. The other argument is an integer of the kind where the
   outcome mentions it. *)
Lemma named_of_outcome_sound (statics : static_env) (vars : env)
    (kind : standard_int) (o : outcome) (other : simple) (d : named)
    (v_other : value) (a r : Z) :
  named_of_outcome kind o other = Some d ->
  eval_simple vars other = Some v_other ->
  match o with
  | Exactly _ | Cannot_simplify => True
  | _ => int_of_value kind v_other = Some a
  end ->
  outcome_holds (int_width kind) o a r ->
  eval_named statics vars d = Some (Value_const (const_of_int kind r)).
Proof.
  intros Hnamed Hother Hint Hholds.
  destruct o; simpl in Hnamed, Hholds; try discriminate;
    injection Hnamed as <-; subst r.
  - (* [Exactly] *)
    reflexivity.
  - (* [The_other_side] *)
    simpl. rewrite Hother.
    apply int_of_value_const in Hint as [-> _]. reflexivity.
  - (* [Negation_of_the_other_side] *)
    cbn [eval_named eval_simples eval_simple]. rewrite Hother. simpl.
    rewrite eval_prim_binary. cbn [eval_binary_primitive].
    rewrite int_of_value_const_of_int
      by (apply zero_in_range, int_width_positive).
    rewrite Hint. reflexivity.
  - (* [Bitwise_not_of_the_other_side] *)
    cbn [eval_named eval_simples eval_simple]. rewrite Hother. simpl.
    rewrite eval_prim_unary. cbn [eval_unary_primitive].
    rewrite Hint. reflexivity.
Qed.

Lemma int_of_simple_spec (kind : standard_int) (s : simple) (i : Z) :
  int_of_simple kind s = Some i ->
  ∃ c : const, s = Const c ∧ int_of_value kind (Value_const c) = Some i.
Proof.
  destruct s as [x | sym | c]; simpl; try discriminate.
  intros Hint. exists c. split; [ reflexivity | exact Hint ].
Qed.

(* A value is the same as itself, where [Phys_equal] has a value. *)
Lemma eval_phys_equal_same (statics : static_env) (v : value)
    (same : bool) :
  eval_phys_equal statics v v = Some same -> same = true.
Proof.
  unfold eval_phys_equal.
  destruct (int_of_value Standard_tagged_immediate v) as [i | ].
  - intros [= <-]. apply Z.eqb_refl.
  - destruct (identity_of statics v) as [s | ]; [ | discriminate ].
    intros [= <-]. apply Pos.eqb_refl.
Qed.

(* The identity of a binary primitive, where it has one, has the value
   of the primitive. *)
Lemma identity_of_binary_sound (statics : static_env) (vars : env)
    (p : binary_primitive) (arg1 arg2 : simple) (d : named) (v : value) :
  identity_of_binary p arg1 arg2 = Some d ->
  eval_named statics vars (Prim (Binary p) [arg1; arg2]) = Some v ->
  eval_named statics vars d = Some v.
Proof.
  intros Hidentity Heval.
  cbn [eval_named eval_simples] in Heval.
  destruct (eval_simple vars arg1) as [v1 | ] eqn:Hv1; [ | discriminate ].
  destruct (eval_simple vars arg2) as [v2 | ] eqn:Hv2; [ | discriminate ].
  simpl in Heval. rewrite eval_prim_binary in Heval.
  destruct p as [array_kind load_kind mut | load_width | eq | kind op
                | kind op | kind behaviour];
    try discriminate;
    cbn [identity_of_binary] in Hidentity;
    cbn [eval_binary_primitive] in Heval.
  - (* [Phys_equal] of a variable and itself *)
    destruct arg1 as [x | | ]; try discriminate.
    destruct arg2 as [y | | ]; try discriminate.
    destruct (Pos.eqb_spec x y) as [<- | _]; [ | discriminate ].
    injection Hidentity as <-.
    simpl in Hv1, Hv2. rewrite Hv1 in Hv2. injection Hv2 as <-.
    destruct (eval_phys_equal statics v1 v1) as [same | ] eqn:Hsame;
      [ | discriminate ].
    apply eval_phys_equal_same in Hsame. subst same.
    injection Heval as <-. destruct eq; reflexivity.
  - (* [Int_arith] *)
    destruct (int_of_value kind v1) as [a | ] eqn:Ha; [ | discriminate ].
    destruct (int_of_value kind v2) as [b | ] eqn:Hb; [ | discriminate ].
    simpl in Heval.
    destruct (eval_int_arith (int_width kind) op a b) as [r | ] eqn:Hr;
      [ | discriminate ].
    injection Heval as <-.
    destruct (int_of_simple kind arg1) as [lhs | ] eqn:Hlhs,
      (int_of_simple kind arg2) as [rhs | ] eqn:Hrhs;
      try discriminate.
    + (* the left argument is known *)
      destruct (is_const arg2); [ discriminate | ].
      destruct (int_of_simple_spec _ _ _ Hlhs) as (c & -> & Hc).
      injection Hv1 as <-. rewrite Hc in Ha. injection Ha as ->.
      apply (named_of_outcome_sound statics vars kind _ arg2 d v2 b r
        Hidentity Hv2).
      * destruct (op_rhs_unknown op a); try exact I; exact Hb.
      * apply op_rhs_unknown_sound;
          [ apply int_width_positive
          | exact (proj2 (int_of_value_const _ _ _ Hb)) | exact Hr ].
    + (* the right argument is known *)
      destruct (is_const arg1); [ discriminate | ].
      destruct (int_of_simple_spec _ _ _ Hrhs) as (c & -> & Hc).
      injection Hv2 as <-. rewrite Hc in Hb. injection Hb as ->.
      apply (named_of_outcome_sound statics vars kind _ arg1 d v1 a r
        Hidentity Hv1).
      * destruct (op_lhs_unknown op b); try exact I; exact Ha.
      * apply op_lhs_unknown_sound;
          [ apply int_width_positive
          | exact (proj2 (int_of_value_const _ _ _ Ha)) | exact Hr ].
  - (* [Int_shift] *)
    destruct (int_of_value kind v1) as [a | ] eqn:Ha; [ | discriminate ].
    destruct (int_of_value Standard_naked_immediate v2) as [amount | ]
      eqn:Hamount;
      [ | discriminate ].
    simpl in Heval.
    destruct ((0 <=? amount) && (amount <? shift_bound kind)) eqn:Hbound;
      [ | discriminate ].
    injection Heval as <-.
    apply andb_true_iff in Hbound as [Hpos _]. apply Z.leb_le in Hpos.
    destruct (int_of_simple kind arg1) as [lhs | ] eqn:Hlhs,
      (int_of_simple Standard_naked_immediate arg2) as [rhs | ] eqn:Hrhs;
      try discriminate.
    + (* the integer shifted is known *)
      destruct (is_const arg2); [ discriminate | ].
      destruct (int_of_simple_spec _ _ _ Hlhs) as (c & -> & Hc).
      injection Hv1 as <-. rewrite Hc in Ha. injection Ha as ->.
      apply (named_of_outcome_sound statics vars kind _ arg2 d v2 amount
        _ Hidentity Hv2).
      * destruct op; simpl;
          repeat
            match goal with
            | |- context [?x =? ?y] => destruct (x =? y)
            end;
          exact I.
      * apply shift_rhs_unknown_sound;
          [ apply int_width_positive | exact Hpos ].
    + (* the amount is known *)
      destruct (is_const arg1); [ discriminate | ].
      destruct (int_of_simple_spec _ _ _ Hrhs) as (c & -> & Hc).
      injection Hv2 as <-. rewrite Hc in Hamount. injection Hamount as ->.
      apply (named_of_outcome_sound statics vars kind _ arg1 d v1 a _
        Hidentity Hv1).
      * unfold shift_lhs_unknown.
        destruct (amount =? 0); [ exact Ha | exact I ].
      * apply shift_lhs_unknown_sound;
          [ apply int_width_positive
          | exact (proj2 (int_of_value_const _ _ _ Ha)) ].
Qed.

Lemma binary_applied_spec (p : prim) (args : list simple)
    (b : binary_primitive) (arg1 arg2 : simple) :
  binary_applied p args = Some (b, arg1, arg2) ->
  p = Binary b ∧ args = [arg1; arg2].
Proof.
  destruct p as [u | b' | variadic | heap | rp]; try discriminate.
  destruct args as [ | a1 [ | a2 [ | a3 args]]]; try discriminate.
  intros [= <- <- <-]. auto.
Qed.

Lemma identity_named_sound (statics : static_env) (vars : env) (d : named)
    (v : value) :
  eval_named statics vars d = Some v ->
  eval_named statics vars (identity_named d) = Some v.
Proof.
  intros Hd. unfold identity_named.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hd.
  destruct (binary_applied p args) as [[[b arg1] arg2] | ] eqn:Happlied;
    [ | exact Hd ].
  destruct (identity_of_binary b arg1 arg2) as [d' | ] eqn:Hidentity;
    [ | exact Hd ].
  destruct (binary_applied_spec _ _ _ _ _ Happlied) as [-> ->].
  exact (identity_of_binary_sound statics vars b arg1 arg2 d' v Hidentity
    Hd).
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* As in FlambdaBlockLoadProofs.v, with nothing to know: at a [Let], what
   replaces the operation has its value or the operation has none; then
   the parts under binders, by the induction. *)
Fixpoint apply_identities_refines_at (e : expr) :
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap),
    refines (State e vars conts statics hp)
      (State (apply_identities e) vars conts statics hp)
with apply_identities_in_handler_refines (h : continuation_handler) :
  ∀ (k : continuation) (body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    refines (State (Let_cont_non_recursive k h body) vars conts statics hp)
      (State
         (Let_cont_non_recursive k (apply_identities_in_handler h) body)
         vars conts statics hp).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros vars conts statics hp; simpl.
    + (* Let *)
      destruct (eval_named statics vars d) as [v | ] eqn:Hd;
        [ destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind | ].
      * eapply refines_step;
          [ rewrite step_let, Hd, Hkind; reflexivity
          | rewrite step_let, (identity_named_sound statics vars d v Hd),
              Hkind;
            reflexivity
          | ].
        apply apply_identities_refines_at.
      * apply step_undefined_refines. rewrite step_let, Hd, Hkind.
        reflexivity.
      * apply step_undefined_refines. rewrite step_let, Hd. reflexivity.
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ]; [ | exact I ].
      apply apply_identities_refines_at.
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k (apply_identities_in_handler h) body)
           vars conts statics hp)).
      * apply apply_identities_in_handler_refines.
      * apply let_cont_non_recursive_refines_step,
          apply_identities_refines_at.
    + (* Let_cont_recursive *)
      apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply apply_identities_refines_at.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros k body vars conts statics hp. simpl.
    apply let_cont_handler_refines_at.
    intros vs handler_vars conts' statics' hp' _ _.
    apply apply_identities_refines_at.
Qed.

Theorem apply_identities_refines : apply_identities_refines_statement.
Proof. intros e vars conts statics. apply apply_identities_refines_at. Qed.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

(* A renaming leaves a constant as it is, and makes no other argument a
   constant. *)
Lemma int_of_simple_renamed (r : renaming) (kind : standard_int)
    (s : simple) :
  int_of_simple kind (rename_simple r s) = int_of_simple kind s.
Proof. destruct s; reflexivity. Qed.

Lemma is_const_renamed (r : renaming) (s : simple) :
  is_const (rename_simple r s) = is_const s.
Proof. destruct s; reflexivity. Qed.

Lemma named_of_outcome_equivariant (r : renaming) (kind : standard_int)
    (o : outcome) (other : simple) :
  named_of_outcome kind o (rename_simple r other)
  = option_map (rename_named r) (named_of_outcome kind o other).
Proof. destruct o; reflexivity. Qed.

(* The two arguments of [Phys_equal] are the same variable after a
   renaming exactly when they were before. *)
Lemma identity_of_binary_equivariant (r : renaming) (p : binary_primitive)
    (arg1 arg2 : simple) :
  identity_of_binary p (rename_simple r arg1) (rename_simple r arg2)
  = option_map (rename_named r) (identity_of_binary p arg1 arg2).
Proof.
  destruct p as [array_kind load_kind mut | width | op | kind op | kind op
                | kind behaviour];
    cbn [identity_of_binary]; try reflexivity.
  - (* [Phys_equal] *)
    destruct arg1 as [x | | ], arg2 as [y | | ]; try reflexivity.
    cbn [rename_simple]. rewrite rename_variable_eqb.
    destruct (Pos.eqb x y); reflexivity.
  - (* [Int_arith] *)
    rewrite !int_of_simple_renamed, !is_const_renamed.
    destruct (int_of_simple kind arg1), (int_of_simple kind arg2);
      try reflexivity.
    + destruct (is_const arg2);
        [ reflexivity | apply named_of_outcome_equivariant ].
    + destruct (is_const arg1);
        [ reflexivity | apply named_of_outcome_equivariant ].
  - (* [Int_shift] *)
    rewrite !int_of_simple_renamed, !is_const_renamed.
    destruct (int_of_simple kind arg1),
      (int_of_simple Standard_naked_immediate arg2);
      try reflexivity.
    + destruct (is_const arg2);
        [ reflexivity | apply named_of_outcome_equivariant ].
    + destruct (is_const arg1);
        [ reflexivity | apply named_of_outcome_equivariant ].
Qed.

Theorem identity_named_equivariant : identity_named_equivariant_statement.
Proof.
  intros r d.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try reflexivity.
  destruct p as [u | b | variadic | heap | rp]; try reflexivity.
  destruct args as [ | arg1 [ | arg2 [ | arg3 args]]]; try reflexivity.
  cbn [rename_named map identity_named binary_applied].
  rewrite identity_of_binary_equivariant.
  destruct (identity_of_binary b arg1 arg2); reflexivity.
Qed.

(* By recursion on the term. A renaming goes into the handlers of
   recursive continuations and into code, where the pass does not: there
   both sides are the term renamed. *)
Fixpoint apply_identities_equivariant_aux (r : renaming) (e : expr) :
  apply_identities (rename_expr r e) = rename_expr r (apply_identities e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - rewrite identity_named_equivariant, apply_identities_equivariant_aux.
    reflexivity.
  - rewrite apply_identities_equivariant_aux. reflexivity.
  - rewrite !apply_identities_equivariant_aux. reflexivity.
  - rewrite apply_identities_equivariant_aux. reflexivity.
Qed.

Theorem apply_identities_equivariant :
  apply_identities_equivariant_statement.
Proof. exact apply_identities_equivariant_aux. Qed.

(* ======================================================================== *)
(* What the pass makes of some operations                                   *)
(* ======================================================================== *)

(* As checks of the definitions against what Simplify does. *)

Module Examples.

Definition x : simple := Var 1%positive.
Definition y : simple := Var 2%positive.
Definition tagged (i : Z) : simple := Const (Tagged_immediate i).
Definition naked (i : Z) : simple := Const (Naked_immediate i).

Definition arith (op : binary_int_arith_op) (a b : simple) : named :=
  Prim (Binary (Int_arith Standard_tagged_immediate op)) [a; b].

Definition shift (op : int_shift_op) (a b : simple) : named :=
  Prim (Binary (Int_shift Standard_tagged_immediate op)) [a; b].

Definition phys_equal (op : equality_comparison) (a b : simple) : named :=
  Prim (Binary (Phys_equal op)) [a; b].

Example add_zero : identity_named (arith Add x (tagged 0)) = Simple x.
Proof. reflexivity. Qed.

Example zero_add : identity_named (arith Add (tagged 0) x) = Simple x.
Proof. reflexivity. Qed.

Example sub_zero : identity_named (arith Sub x (tagged 0)) = Simple x.
Proof. reflexivity. Qed.

Example zero_sub :
  identity_named (arith Sub (tagged 0) x) = arith Sub (tagged 0) x.
Proof. reflexivity. Qed.

Example mul_zero :
  identity_named (arith Mul x (tagged 0)) = Simple (tagged 0).
Proof. reflexivity. Qed.

Example one_mul : identity_named (arith Mul (tagged 1) x) = Simple x.
Proof. reflexivity. Qed.

Example mul_minus_one :
  identity_named (arith Mul x (tagged (-1))) = arith Sub (tagged 0) x.
Proof. reflexivity. Qed.

Example div_one :
  identity_named (arith (Div Unsigned) x (tagged 1)) = Simple x.
Proof. reflexivity. Qed.

Example div_minus_one :
  identity_named (arith (Div Signed) x (tagged (-1)))
  = arith Sub (tagged 0) x.
Proof. reflexivity. Qed.

Example unsigned_div_minus_one_stays :
  identity_named (arith (Div Unsigned) x (tagged (-1)))
  = arith (Div Unsigned) x (tagged (-1)).
Proof. reflexivity. Qed.

Example div_zero_stays :
  identity_named (arith (Div Signed) x (tagged 0))
  = arith (Div Signed) x (tagged 0).
Proof. reflexivity. Qed.

Example one_div_stays :
  identity_named (arith (Div Signed) (tagged 1) x)
  = arith (Div Signed) (tagged 1) x.
Proof. reflexivity. Qed.

Example mod_one :
  identity_named (arith (Mod Signed) x (tagged 1)) = Simple (tagged 0).
Proof. reflexivity. Qed.

Example and_minus_one :
  identity_named (arith And x (tagged (-1))) = Simple x.
Proof. reflexivity. Qed.

Example or_minus_one :
  identity_named (arith Or (tagged (-1)) x) = Simple (tagged (-1)).
Proof. reflexivity. Qed.

Example xor_minus_one :
  identity_named (arith Xor x (tagged (-1)))
  = Prim (Unary (Unary_int_arith Standard_tagged_immediate Not)) [x].
Proof. reflexivity. Qed.

Example shift_by_zero :
  identity_named (shift Lsl x (naked 0)) = Simple x.
Proof. reflexivity. Qed.

Example shift_of_zero :
  identity_named (shift Lsr (tagged 0) x) = Simple (tagged 0).
Proof. reflexivity. Qed.

Example asr_of_minus_one :
  identity_named (shift Asr (tagged (-1)) x) = Simple (tagged (-1)).
Proof. reflexivity. Qed.

Example lsl_of_minus_one_stays :
  identity_named (shift Lsl (tagged (-1)) x) = shift Lsl (tagged (-1)) x.
Proof. reflexivity. Qed.

Example phys_eq_of_a_variable_and_itself :
  identity_named (phys_equal Phys_eq x x) = Simple (naked 1).
Proof. reflexivity. Qed.

Example phys_neq_of_a_variable_and_itself :
  identity_named (phys_equal Phys_neq x x) = Simple (naked 0).
Proof. reflexivity. Qed.

Example phys_eq_of_two_variables_stays :
  identity_named (phys_equal Phys_eq x y) = phys_equal Phys_eq x y.
Proof. reflexivity. Qed.

(* What Simplify has no identity for stays. *)
Example sub_of_itself_stays :
  identity_named (arith Sub x x) = arith Sub x x.
Proof. reflexivity. Qed.

Example or_of_itself_stays : identity_named (arith Or x x) = arith Or x x.
Proof. reflexivity. Qed.

(* An operation on two constants is for the folding of constants. *)
Example two_constants_stay :
  identity_named (arith Add (tagged 0) (tagged 5))
  = arith Add (tagged 0) (tagged 5).
Proof. reflexivity. Qed.

(* A zero of another kind is not the zero of the operation. *)
Example zero_of_another_kind_stays :
  identity_named (arith Add x (naked 0)) = arith Add x (naked 0).
Proof. reflexivity. Qed.

(* The pass goes under a [Let] and into the handler of a non-recursive
   continuation. *)
Example under_binders :
  apply_identities
    (Let_cont_non_recursive 1%positive
       (Handler [1%positive] (make_handler_metadata [Value])
          (Let 2%positive Value (arith Add x (tagged 0)) Invalid))
       (Let 3%positive Value (arith Mul (tagged 1) x) Invalid))
  = Let_cont_non_recursive 1%positive
      (Handler [1%positive] (make_handler_metadata [Value])
         (Let 2%positive Value (Simple x) Invalid))
      (Let 3%positive Value (Simple x) Invalid).
Proof. reflexivity. Qed.

End Examples.
