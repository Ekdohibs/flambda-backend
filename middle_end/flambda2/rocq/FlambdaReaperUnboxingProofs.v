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

(* The proofs of FlambdaReaperUnboxingSpec.v: the checks of
   FlambdaReaperUnboxing.v imply what they are said to mean, and the
   statements of the counterexample are computed. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaReaperGraphSpec
  FlambdaReaperSolutionProofs FlambdaReaperUnboxing
  FlambdaReaperUnboxingSpec.
Import ListNotations.

Lemma node_eqb_false : ∀ n n' : node, n <> n' -> node_eqb n n' = false.
Proof.
  intros n n' Hne. destruct (node_eqb n n') eqn:Heq; [ | reflexivity ].
  apply node_eqb_eq in Heq. contradiction.
Qed.

Lemma field_eqb_decides : decides_equality field_eqb.
Proof.
  intros f f'. split; [ apply field_eqb_eq | intros <-; apply field_eqb_refl ].
Qed.

(* ======================================================================== *)
(* Allocation points                                                        *)
(* ======================================================================== *)

Lemma multiple_allocation_points_iff : ∀ (T : tables) (x : node),
  multiple_allocation_points T x = true
  <-> multiple_allocation_points_of (solution_of_tables T) x.
Proof.
  intros T x.
  unfold multiple_allocation_points, multiple_allocation_points_of.
  cbn [solution_of_tables s_any_source s_sources s_has_source].
  rewrite orb_true_iff, existsb_exists. split.
  - intros [H | ([x' y] & Hin & Htest)]; [ left; exact H | right ].
    apply andb_true_iff in Htest as [Htest Hz].
    apply andb_true_iff in Htest as [Hx Hy].
    apply node_eqb_eq in Hx. subst x'.
    apply existsb_exists in Hz as ([x'' z] & Hin' & Htest').
    apply andb_true_iff in Htest' as [Htest' Hne].
    apply andb_true_iff in Htest' as [Hx Hhz].
    apply node_eqb_eq in Hx. subst x''.
    exists y, z.
    split; [ apply mem_pair_In; exact Hin | split; [ exact Hy | ] ].
    split; [ apply mem_pair_In; exact Hin' | split; [ exact Hhz | ] ].
    intros Heq. subst z. rewrite node_eqb_refl in Hne. discriminate.
  - intros [H | (y & z & Hy & Hhy & Hz & Hhz & Hne)]; [ left; exact H | ].
    right. exists (x, y). split; [ apply mem_pair_In; exact Hy | ].
    rewrite node_eqb_refl, Hhy. cbn.
    apply existsb_exists. exists (x, z).
    split; [ apply mem_pair_In; exact Hz | ].
    rewrite node_eqb_refl, Hhz, node_eqb_false by exact Hne. reflexivity.
Qed.

Theorem check_dominated_sound : check_dominated_sound_statement.
Proof.
  intros T dominated Hcheck a x. unfold check_dominated in Hcheck.
  apply andb_true_iff in Hcheck as [Hclosed Hjust].
  unfold dominated_by_allocation_point.
  cbn [solution_of_tables s_sources s_has_source].
  split.
  - intros Hmem. apply mem_pair_In in Hmem.
    rewrite forallb_forall in Hjust. specialize (Hjust _ Hmem).
    cbn beta iota in Hjust.
    apply andb_true_iff in Hjust as [Hjust Hm].
    apply andb_true_iff in Hjust as [Hs Hh].
    split; [ exact Hs | split; [ exact Hh | ] ].
    rewrite <- multiple_allocation_points_iff.
    apply negb_true_iff in Hm. rewrite Hm. discriminate.
  - intros (Hs & Hh & Hm). rewrite forallb_forall in Hclosed.
    assert (Hin : (a, x) ∈ derive_dominated T).
    { unfold derive_dominated. apply in_flat_map. exists (x, a).
      split; [ apply mem_pair_In; exact Hs | ].
      rewrite Hh. destruct (multiple_allocation_points T x) eqn:E.
      - exfalso. apply Hm. apply multiple_allocation_points_iff. exact E.
      - left. reflexivity. }
    exact (Hclosed _ Hin).
Qed.

(* ======================================================================== *)
(* The rules of the unboxing                                                *)
(* ======================================================================== *)

(* Simplify a test taken at a tuple: the match on the tuple, the equality
   tests of a node or a field with itself, and every test whose answer a
   hypothesis gives. *)
Ltac simplify_pass H :=
  cbn beta iota in H;
  rewrite ?node_eqb_refl, ?field_eqb_refl in H;
  rewrite_known H;
  cbn [negb andb orb] in H.

(* [H] is a test that passes: take each [forallb] at a member of its list
   that the context has, trying each such member in turn, until what is
   left is the goal. *)
Ltac passes H :=
  simplify_pass H;
  first
    [ exact H
    | lazymatch type of H with
      | forallb _ ?l = true =>
          match goal with
          | Hin : ?t ∈ l |- _ =>
              let H' := fresh "H" in
              pose proof H as H'; rewrite forallb_forall in H';
              specialize (H' t Hin); passes H'
          end
      end ].

(* The goal is a field of [unboxing_closed], and [c] the test of that
   rule. *)
Ltac solve_rule c :=
  intros;
  match goal with
  | Hc : ?t = true |- _ =>
      lazymatch t with context [c] => idtac end;
      let H := fresh "H" in
      pose proof Hc as H; unfold c in H;
      unfold_relations; cbv beta in *;
      repeat match goal with
      | Hne : ?a <> ?b |- _ =>
          lazymatch type of a with
          | node => apply node_eqb_false in Hne
          end
      end;
      add_memberships; passes H
  end.

Section Closed.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Variable cannot_change : list node.
Variable max_unbox_size : nat.
Variable lambda_lifting : bool.
Variable R : unboxing_tables.

(* The rule of [max_unbox_size]: fields that are distinct and used are as
   many as [used_fields] has at most. *)
Lemma used_fields_length : ∀ (x : node) (fields : list field),
  NoDup fields ->
  (∀ f : field,
     In f fields ->
     s_field_used (solution_of_tables T) x f = true
     ∧ is_real_field f = true ∧ is_function_slot f = false) ->
  length fields <= length (used_fields T x).
Proof.
  intros x fields Hnodup Hall. apply NoDup_incl_length; [ exact Hnodup | ].
  intros f Hf. destruct (Hall f Hf) as (Hused & Hreal & Hslot).
  cbn [solution_of_tables s_field_used] in Hused.
  apply mem_node_field_In in Hused.
  assert (Hin : f ∈ flat_map
                       (fun '(y, f) =>
                          if node_eqb y x && is_real_field f
                             && negb (is_function_slot f)
                          then [f]
                          else [])
                       (t_field_used T)).
  { apply in_flat_map. exists (x, f). split; [ exact Hused | ].
    rewrite node_eqb_refl, Hreal, Hslot. left. reflexivity. }
  unfold used_fields.
  assert (Hamong : Keyed.among field_eqb f
                     (Keyed.dedup field_eqb
                        (flat_map
                           (fun '(y, f) =>
                              if node_eqb y x && is_real_field f
                                 && negb (is_function_slot f)
                              then [f]
                              else [])
                           (t_field_used T))) = true).
  { rewrite (keyed_among_dedup field _ field_eqb_decides).
    unfold Keyed.among. apply existsb_exists. exists f.
    split; [ exact Hin | apply field_eqb_refl ]. }
  unfold Keyed.among in Hamong. apply existsb_exists in Hamong.
  destruct Hamong as (f' & Hin' & Heq). apply field_eqb_eq in Heq.
  subst f'. exact Hin'.
Qed.

Lemma check_unboxing_closed_sound :
  check_unboxing_closed is_local F T cannot_change max_unbox_size
    lambda_lifting R = true ->
  unboxing_closed is_local (graph_of_facts F) (solution_of_tables T)
    (fun n => mem_node n cannot_change) max_unbox_size lambda_lifting
    (fun n => mem_node n (ut_crr0 R)) (fun n => mem_node n (ut_crr1 R))
    (fun n => mem_node n (ut_crr R)) (fun n => mem_node n (ut_cu0 R))
    (fun n => mem_node n (ut_cu R)).
Proof.
  intros Hcheck. unfold check_unboxing_closed in Hcheck.
  repeat match goal with
  | H : _ && _ = true |- _ => apply andb_true_iff in H as [? ?]
  end.
  constructor.
  - solve_rule check_crr0_any_usage.
  - solve_rule check_crr0_local_reading.
  - solve_rule check_crr0_local_sources.
  - solve_rule check_crr0_other_source.
  - solve_rule check_crr0_any_source_usage.
  - solve_rule check_crr0_other_own_source.
  - solve_rule check_crr0_not_own_source.
  - solve_rule check_crr0_any_source.
  - solve_rule check_crr0_call_witness.
  - solve_rule check_crr1_crr0.
  - solve_rule check_crr1_function_slot.
  - solve_rule check_crr_crr1.
  - solve_rule check_crr_block_like.
  - solve_rule check_cu0_cannot_change_representation.
  - solve_rule check_cu0_any_usage.
  - solve_rule check_cu0_called_cannot_change.
  - solve_rule check_cu0_called.
  - solve_rule check_cu0_result.
  - solve_rule check_cu0_parameter.
  - solve_rule check_cu0_symbol.
  - solve_rule check_cu0_stored.
  - (* [cu0_max_unbox_size] *)
    intros x fields Hnodup Hall Hmore.
    pose proof (used_fields_length x fields Hnodup Hall) as Hlength.
    destruct fields as [ | f fields ]; [ cbn in Hmore; lia | ].
    destruct (Hall f (or_introl eq_refl)) as (Hused & _ & _).
    cbn [solution_of_tables s_field_used] in Hused.
    apply mem_node_field_In in Hused.
    match goal with
    | Hc : ?t = true |- _ =>
        lazymatch t with context [check_cu0_max_unbox_size] => idtac end;
        unfold check_cu0_max_unbox_size in Hc;
        rewrite forallb_forall in Hc; specialize (Hc _ Hused);
        cbn beta iota in Hc;
        replace (Nat.ltb max_unbox_size (length (used_fields T x))) with true
          in Hc by (symmetry; apply Nat.ltb_lt; lia);
        exact Hc
    end.
  - solve_rule check_cu_cu0.
  - solve_rule check_cu_function_slot.
Qed.

End Closed.

Theorem check_to_unbox_sound : check_to_unbox_sound_statement.
Proof.
  intros is_local F T cannot_change max_unbox_size lambda_lifting to_unbox
    Hcheck.
  unfold check_to_unbox in Hcheck.
  set (R := unboxing_relations is_local F T cannot_change max_unbox_size
              lambda_lifting) in Hcheck.
  apply andb_true_iff in Hcheck as [Hcheck Hcomplete].
  apply andb_true_iff in Hcheck as [Hclosed Hto].
  exists (fun n => mem_node n (ut_crr0 R)), (fun n => mem_node n (ut_crr1 R)),
    (fun n => mem_node n (ut_crr R)), (fun n => mem_node n (ut_cu0 R)),
    (fun n => mem_node n (ut_cu R)).
  split; [ | split ].
  - apply check_unboxing_closed_sound. exact Hclosed.
  - intros x Hx. apply mem_node_In in Hx. rewrite forallb_forall in Hto.
    specialize (Hto x Hx). apply andb_true_iff in Hto as [Husage Hcu].
    split; [ exact Husage | apply negb_true_iff; exact Hcu ].
  - intros x Husage Hsource Hcu.
    cbn [solution_of_tables s_has_usage s_has_source] in Husage, Hsource.
    apply mem_node_In in Husage. rewrite forallb_forall in Hcomplete.
    specialize (Hcomplete x Husage). cbn beta in Hcu.
    rewrite Hsource, Hcu in Hcomplete. exact Hcomplete.
Qed.

Theorem check_to_change_representation_sound :
  check_to_change_representation_sound_statement.
Proof.
  intros is_local F T cannot_change max_unbox_size lambda_lifting to_unbox
    tcr Hcheck.
  unfold check_to_change_representation, check_to_unbox in Hcheck.
  cbv zeta in Hcheck.
  set (R := unboxing_relations is_local F T cannot_change max_unbox_size
              lambda_lifting) in Hcheck.
  apply andb_true_iff in Hcheck as [Hcheck Hcomplete'].
  apply andb_true_iff in Hcheck as [Hcheck Hto'].
  apply andb_true_iff in Hcheck as [Hcheck Hcomplete].
  apply andb_true_iff in Hcheck as [Hclosed Hto].
  exists (fun n => mem_node n (ut_crr0 R)), (fun n => mem_node n (ut_crr1 R)),
    (fun n => mem_node n (ut_crr R)), (fun n => mem_node n (ut_cu0 R)),
    (fun n => mem_node n (ut_cu R)).
  split; [ | split; [ | split; [ | split ] ] ].
  - apply check_unboxing_closed_sound. exact Hclosed.
  - intros x Hx. apply mem_node_In in Hx. rewrite forallb_forall in Hto.
    specialize (Hto x Hx). apply andb_true_iff in Hto as [Husage Hcu].
    split; [ exact Husage | apply negb_true_iff; exact Hcu ].
  - intros x Husage Hsource Hcu.
    cbn [solution_of_tables s_has_usage s_has_source] in Husage, Hsource.
    apply mem_node_In in Husage. rewrite forallb_forall in Hcomplete.
    specialize (Hcomplete x Husage). cbn beta in Hcu.
    rewrite Hsource, Hcu in Hcomplete. exact Hcomplete.
  - intros x Hx. apply mem_node_In in Hx. rewrite forallb_forall in Hto'.
    specialize (Hto' x Hx).
    apply andb_true_iff in Hto' as [Hto' Hunbox].
    apply andb_true_iff in Hto' as [Husage Hcrr].
    split; [ exact Husage | ].
    split; apply negb_true_iff; assumption.
  - intros x Husage Hsource Hcrr Hunbox.
    cbn [solution_of_tables s_has_usage s_has_source] in Husage, Hsource.
    apply mem_node_In in Husage. rewrite forallb_forall in Hcomplete'.
    specialize (Hcomplete' x Husage). cbn beta in Hcrr.
    rewrite Hsource, Hcrr, Hunbox in Hcomplete'. exact Hcomplete'.
Qed.

Theorem check_sets_whole_sound : check_sets_whole_sound_statement.
Proof.
  intros F T to_unbox tcr Hcheck x y f Hin Hslot Hx Hy.
  unfold check_sets_whole in Hcheck. rewrite forallb_forall in Hcheck.
  specialize (Hcheck _ Hin). cbn beta iota in Hcheck.
  cbn [solution_of_tables s_has_usage] in Hx, Hy.
  rewrite Hslot, Hx, Hy in Hcheck. cbn [negb orb] in Hcheck.
  apply andb_true_iff in Hcheck as [Hunbox Htcr].
  apply Bool.eqb_prop in Hunbox. apply Bool.eqb_prop in Htcr.
  split; assumption.
Qed.

(* ======================================================================== *)
(* A counterexample: lambda lifting and a dead callee                       *)
(* ======================================================================== *)

Theorem lambda_lifting_dead_callee : lambda_lifting_dead_callee_statement.
Proof.
  split; [ intros [ | ]; vm_compute; auto 10 | ].
  split; [ exists 20; vm_compute; reflexivity | ].
  split; [ exists 20; vm_compute; reflexivity | ].
  exists
    (Code [40] 41 3 4
       (make_code_metadata [Unboxed_product [Singleton Value]]
          [Unboxed_product [Singleton Value]])
       (Apply_cont
          {| ac_continuation := 3; ac_args := [Var 40];
             ac_trap_action := None |}))%positive.
  split; [ vm_compute; reflexivity | exists 20; vm_compute; reflexivity ].
Qed.

Theorem lambda_lifting_dead_callee_cause :
  lambda_lifting_dead_callee_cause_statement.
Proof.
  do 6 (split; [ vm_compute; reflexivity | ]).
  split; [ exists 20; vm_compute; reflexivity | vm_compute; reflexivity ].
Qed.

(* ======================================================================== *)
(* A counterexample: a result kept by a continuation and unboxed by code    *)
(* ======================================================================== *)

Theorem kept_result_unboxed : kept_result_unboxed_statement.
Proof.
  split; [ intros [ | ]; vm_compute; auto 10 | ].
  split; [ exists 20; vm_compute; reflexivity | ].
  split; [ exists 20; vm_compute; reflexivity | ].
  split; [ exists 20; vm_compute; reflexivity | ].
  vm_compute. auto 10.
Qed.

Theorem kept_result_unboxed_cause : kept_result_unboxed_cause_statement.
Proof. vm_compute. auto 10. Qed.

(* ======================================================================== *)
(* A fixed miscompile: a result unboxed by code and deleted by a            *)
(* continuation                                                             *)
(* ======================================================================== *)

Theorem unbox_dead : unbox_dead_statement.
Proof.
  do 12 (split; [ vm_compute; reflexivity | ]).
  split; [ exists 20; vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exists 20; vm_compute; reflexivity | ].
  do 2 (split; [ vm_compute; reflexivity | ]).
  exists 20; vm_compute; reflexivity.
Qed.
