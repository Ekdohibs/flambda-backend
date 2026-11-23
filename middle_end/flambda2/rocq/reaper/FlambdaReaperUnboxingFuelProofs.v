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

(* The proofs of FlambdaReaperUnboxingFuelSpec.v. Nothing here needs
   review.

   [mk_as_with]: [mk_unboxed_fields] at a depth is
   [mk_unboxed_fields_with] whose recursive call is the depth below
   ([nest_depth]); [nest_levels] and [unboxing_nest] are such calls at
   depths that they do not compute ([levels_depth],
   [unboxing_nest_depth]). [with_ext_calls]: [mk_unboxed_fields_with]
   reads its recursive call only at the nested calls. The bound: [reach]
   is a chain of nested calls; where the recursion ends no call reaches
   the one it is made from ([ends_no_cycle]); every call is one of
   [children], of length [unboxing_depth T] ([call_in_children],
   [children_length]); so the calls of a chain, which reach each other,
   are distinct, and no more than [children] ([within_from]). The
   converse: where the depths [n] and [S n] give the same unboxing, no
   chain is longer than [n] ([agree_within]): the field of a nested call
   is a variable at the depth 0, and unboxed at the depth below
   otherwise; the fields after it start from the same name, which the
   number of variables gives ([mk_next]).

   For the proofs of the other files, which read the model's unboxings
   as [mk_unboxed_fields] at a depth: [name_unboxed_fields_depth] and
   [changed_representations_depth]. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperUnboxing
  FlambdaReaperUnboxingFuelSpec.
From OxCamlFlambda2 Require FlambdaReaperLeast.
From OxCamlFlambda2 Require Import FlambdaReaperSolutionProofs.
Import ListNotations.

Local Abbreviation nest_type :=
  (node -> list (field * field_use) -> positive
   -> option (unboxed_fields * positive)).

Lemma node_eqb_decides : decides_equality node_eqb.
Proof.
  intros n n'. split; [ apply node_eqb_eq | intros <-; apply node_eqb_refl ].
Qed.

Lemma dedup_in (l : list node) (x : node) :
  x ∈ Keyed.dedup node_eqb l -> x ∈ l.
Proof.
  rewrite <- !(keyed_among_in node node_eqb node_eqb_decides).
  rewrite (keyed_among_dedup node node_eqb node_eqb_decides).
  exact (fun H => H).
Qed.

Section Fuel.

Variable F : facts.
Variable T : tables.
Variable cm : finite_map variable.
Variable dominated : list (node * node).
Variable to_unbox : list node.

(* ======================================================================== *)
(* The depth                                                                *)
(* ======================================================================== *)

(* The recursive call of [mk_unboxed_fields] at the depth [fuel]. *)
Definition nest_depth (fuel : nat) : nest_type :=
  fun block fields next =>
    match fuel with
    | O => None
    | S fuel =>
      Some (mk_unboxed_fields F T cm dominated to_unbox fuel block fields
              next)
    end.

(* One field of [mk_unboxed_fields_with]. *)
Definition step_with (nest : nest_type) (block : node) (f : field)
    (use : field_use) (next : positive) : option unboxed_field * positive :=
  if negb (is_data_field f)
  then (None, next)
  else
    match get_single_field_source F T block f with
    | No_source => (None, next)
    | field_source =>
      let default := (Some (Not_unboxed next), Pos.succ next) in
      match use with
      | Use_unknown => default
      | Use_known flow_to =>
        if forallb (has_to_be_unboxed dominated to_unbox) flow_to
        then
          match field_source with
          | One_source source =>
            match nest source (fields_read F T cm flow_to) next with
            | Some (nested, next) => (Some (Unboxed nested), next)
            | None => default
            end
          | _ => default
          end
        else default
      end
    end.

Lemma with_nil (nest : nest_type) (block : node) (next : positive) :
  mk_unboxed_fields_with F T cm dominated to_unbox nest block [] next
  = ([], next).
Proof. reflexivity. Qed.

Lemma with_cons (nest : nest_type) (block : node) (f : field)
    (use : field_use) (fields : list (field * field_use)) (next : positive) :
  mk_unboxed_fields_with F T cm dominated to_unbox nest block
    ((f, use) :: fields) next
  = let '(u, next) := step_with nest block f use next in
    let '(rest, next) :=
      mk_unboxed_fields_with F T cm dominated to_unbox nest block fields next
    in
    (match u with
     | Some u => (f, u) :: rest
     | None => rest
     end,
     next).
Proof. reflexivity. Qed.

Lemma mk_nil (fuel : nat) (block : node) (next : positive) :
  mk_unboxed_fields F T cm dominated to_unbox fuel block [] next = ([], next).
Proof. destruct fuel; reflexivity. Qed.

Lemma mk_cons (fuel : nat) (block : node) (f : field) (use : field_use)
    (fields : list (field * field_use)) (next : positive) :
  mk_unboxed_fields F T cm dominated to_unbox fuel block ((f, use) :: fields)
    next
  = let '(u, next) := step_with (nest_depth fuel) block f use next in
    let '(rest, next) :=
      mk_unboxed_fields F T cm dominated to_unbox fuel block fields next
    in
    (match u with
     | Some u => (f, u) :: rest
     | None => rest
     end,
     next).
Proof. destruct fuel; reflexivity. Qed.

Lemma mk_as_with (fuel : nat) (block : node) :
  ∀ (fields : list (field * field_use)) (next : positive),
    mk_unboxed_fields F T cm dominated to_unbox fuel block fields next
    = mk_unboxed_fields_with F T cm dominated to_unbox (nest_depth fuel)
        block fields next.
Proof.
  induction fields as [ | [f use] fields IH]; intros next;
    [ rewrite mk_nil; reflexivity | ].
  rewrite mk_cons, with_cons.
  destruct (step_with (nest_depth fuel) block f use next) as [u n].
  rewrite IH. reflexivity.
Qed.

Lemma nested_call_tail (block : node) (f : field) (use : field_use)
    (fields : list (field * field_use)) (source : node)
    (fields' : list (field * field_use)) :
  nested_call F T cm dominated to_unbox block fields source fields' ->
  nested_call F T cm dominated to_unbox block ((f, use) :: fields) source
    fields'.
Proof.
  intros Hc. destruct Hc as [f' flow_to source Hin Hd Hs Hall].
  exact (Nested_call F T cm dominated to_unbox block ((f, use) :: fields) f'
           flow_to source (or_intror Hin) Hd Hs Hall).
Qed.

Lemma within_unfold (n : nat) (block : node)
    (fields : list (field * field_use)) :
  calls_within F T cm dominated to_unbox n block fields
  <-> ∀ (source : node) (fields' : list (field * field_use)),
        nested_call F T cm dominated to_unbox block fields source fields' ->
        match n with
        | O => False
        | S n => calls_within F T cm dominated to_unbox n source fields'
        end.
Proof. destruct n; reflexivity. Qed.

Lemma within_tail (n : nat) (block : node) (f : field) (use : field_use)
    (fields : list (field * field_use)) :
  calls_within F T cm dominated to_unbox n block ((f, use) :: fields) ->
  calls_within F T cm dominated to_unbox n block fields.
Proof.
  rewrite !within_unfold. intros Hw source fields' Hc.
  exact (Hw source fields' (nested_call_tail block f use fields _ _ Hc)).
Qed.

(* [mk_unboxed_fields_with] reads [nest] only at the nested calls. *)
Lemma with_ext_calls (nest1 nest2 : nest_type) (block : node) :
  ∀ fields : list (field * field_use),
    (∀ (source : node) (fields' : list (field * field_use))
       (next : positive),
       nested_call F T cm dominated to_unbox block fields source fields' ->
       nest1 source fields' next = nest2 source fields' next) ->
    ∀ next : positive,
      mk_unboxed_fields_with F T cm dominated to_unbox nest1 block fields
        next
      = mk_unboxed_fields_with F T cm dominated to_unbox nest2 block fields
          next.
Proof.
  induction fields as [ | [f use] fields IH]; intros Hn next;
    [ reflexivity | ].
  rewrite !with_cons.
  assert (Hs : step_with nest1 block f use next
               = step_with nest2 block f use next).
  { unfold step_with.
    destruct (is_data_field f) eqn:Hd; [ | reflexivity ]. cbn [negb].
    destruct (get_single_field_source F T block f) as [ | source | ]
      eqn:Hsrc; try reflexivity.
    destruct use as [ | flow_to]; [ reflexivity | ].
    destruct (forallb (has_to_be_unboxed dominated to_unbox) flow_to)
      eqn:Hall; [ | reflexivity ].
    rewrite (Hn source _ next); [ reflexivity | ].
    exact (Nested_call F T cm dominated to_unbox block
             ((f, Use_known flow_to) :: fields) f flow_to source
             (or_introl eq_refl) Hd Hsrc Hall). }
  rewrite Hs. destruct (step_with nest2 block f use next) as [u n].
  rewrite (IH (fun s fs' n' Hc =>
                 Hn s fs' n' (nested_call_tail block f use fields s fs' Hc))
             n).
  reflexivity.
Qed.

Lemma with_ext (nest1 nest2 : nest_type) :
  (∀ (source : node) (fields : list (field * field_use)) (next : positive),
     nest1 source fields next = nest2 source fields next) ->
  ∀ (block : node) (fields : list (field * field_use)) (next : positive),
    mk_unboxed_fields_with F T cm dominated to_unbox nest1 block fields next
    = mk_unboxed_fields_with F T cm dominated to_unbox nest2 block fields
        next.
Proof.
  intros Hn block fields.
  apply with_ext_calls. intros s fs' n _. apply Hn.
Qed.

Lemma levels_depth (l : list (node * node)) :
  ∀ (nest : nest_type) (k : nat),
    (∀ (source : node) (fields : list (field * field_use)) (next : positive),
       nest source fields next = nest_depth k source fields next) ->
    ∀ (source : node) (fields : list (field * field_use)) (next : positive),
      nest_levels F T cm dominated to_unbox l nest source fields next
      = nest_depth (2 ^ length l + k) source fields next.
Proof.
  induction l as [ | p l IH]; intros nest k Hnest source fields next.
  - cbn [nest_levels length Nat.pow Nat.mul Nat.add]. unfold nest_depth.
    rewrite mk_as_with. f_equal. apply with_ext. exact Hnest.
  - cbn [nest_levels].
    rewrite (IH (nest_levels F T cm dominated to_unbox l nest)
               (2 ^ length l + k)); [ | intros; apply IH; exact Hnest ].
    replace (2 ^ length (p :: l) + k) with (2 ^ length l + (2 ^ length l + k));
      [ reflexivity | ].
    cbn [length]. rewrite Nat.pow_succ_r'. lia.
Qed.

Lemma unboxing_nest_depth (k : list (node * node)) :
  ∀ (source : node) (fields : list (field * field_use)) (next : positive),
    unboxing_nest F T cm dominated to_unbox k source fields next
    = nest_depth (length k * 2 ^ length (t_usages T)) source fields next.
Proof.
  induction k as [ | p k IH]; intros source fields next; [ reflexivity | ].
  cbn [unboxing_nest].
  rewrite (levels_depth (t_usages T)
             (unboxing_nest F T cm dominated to_unbox k)
             (length k * 2 ^ length (t_usages T)) IH).
  replace (length (p :: k) * 2 ^ length (t_usages T))
    with (2 ^ length (t_usages T) + length k * 2 ^ length (t_usages T));
    [ reflexivity | ].
  cbn [length]. rewrite Nat.mul_succ_l. lia.
Qed.

Lemma mk_unboxed_fields_full_depth (block : node)
    (fields : list (field * field_use)) (next : positive) :
  mk_unboxed_fields_full F T cm dominated to_unbox block fields next
  = mk_unboxed_fields F T cm dominated to_unbox (unboxing_depth T) block
      fields next.
Proof.
  unfold mk_unboxed_fields_full, unboxing_depth. rewrite mk_as_with.
  apply with_ext. exact (unboxing_nest_depth (t_sources T)).
Qed.

(* ======================================================================== *)
(* The relation                                                             *)
(* ======================================================================== *)

Lemma within_stable (n : nat) :
  ∀ (block : node) (fields : list (field * field_use)),
    calls_within F T cm dominated to_unbox n block fields ->
    ∀ m : nat, n <= m ->
    ∀ next : positive,
      mk_unboxed_fields F T cm dominated to_unbox m block fields next
      = mk_unboxed_fields F T cm dominated to_unbox n block fields next.
Proof.
  induction n as [ | n IH]; intros block fields Hw m Hle next;
    rewrite !mk_as_with; apply with_ext_calls; intros s fs' nx Hc;
    apply within_unfold in Hw; specialize (Hw s fs' Hc).
  - destruct Hw.
  - destruct m as [ | m]; [ lia | ]. unfold nest_depth. f_equal.
    apply IH; [ exact Hw | lia ].
Qed.

Lemma unboxes_mk (block : node) (fields : list (field * field_use))
    (next : positive) (r : unboxed_fields * positive) :
  unboxes F T cm dominated to_unbox block fields next r ->
  ∀ n : nat,
    calls_within F T cm dominated to_unbox n block fields ->
    mk_unboxed_fields F T cm dominated to_unbox n block fields next = r.
Proof.
  induction 1 as
      [ block next
      | block f use fields next r Hout H IH
      | block f fields next rest next' Hd Hs H IH
      | block f flow_to fields next rest next' Hd Hs Hall H IH
      | block f flow_to fields next rest next' Hd Hs Hall H IH
      | block f flow_to source fields next nested next1 rest next2 Hd Hs Hall
          H1 IH1 H2 IH2 ];
    intros n Hw; [ apply mk_nil | | | | | ];
    rewrite mk_cons; pose proof (within_tail n block f _ fields Hw) as Hw'.
  - assert (Hst : step_with (nest_depth n) block f use next = (None, next)).
    { unfold step_with. destruct Hout as [Hd | Hs]; [ rewrite Hd; reflexivity
                                                     | ].
      destruct (is_data_field f); [ | reflexivity ]. cbn [negb].
      rewrite Hs. reflexivity. }
    rewrite Hst, (IH n Hw'). destruct r. reflexivity.
  - assert (Hst : step_with (nest_depth n) block f Use_unknown next
                  = (Some (Not_unboxed next), Pos.succ next)).
    { unfold step_with. rewrite Hd. cbn [negb].
      destruct (get_single_field_source F T block f);
        [ contradiction | reflexivity | reflexivity ]. }
    rewrite Hst, (IH n Hw'). reflexivity.
  - assert (Hst : step_with (nest_depth n) block f (Use_known flow_to) next
                  = (Some (Not_unboxed next), Pos.succ next)).
    { unfold step_with. rewrite Hd, Hall. cbn [negb].
      destruct (get_single_field_source F T block f);
        [ contradiction | reflexivity | reflexivity ]. }
    rewrite Hst, (IH n Hw'). reflexivity.
  - assert (Hst : step_with (nest_depth n) block f (Use_known flow_to) next
                  = (Some (Not_unboxed next), Pos.succ next)).
    { unfold step_with. rewrite Hd, Hs, Hall. reflexivity. }
    rewrite Hst, (IH n Hw'). reflexivity.
  - pose proof (proj1 (within_unfold _ _ _) Hw) as Hw2.
    pose proof (Hw2 source _
                  (Nested_call F T cm dominated to_unbox block
                     ((f, Use_known flow_to) :: fields) f flow_to source
                     (or_introl eq_refl) Hd Hs Hall))
      as Hn.
    destruct n as [ | n]; [ destruct Hn | ].
    assert (Hst : step_with (nest_depth (S n)) block f (Use_known flow_to) next
                  = (Some (Unboxed nested), next1)).
    { unfold step_with. rewrite Hd, Hs, Hall. cbn [negb]. unfold nest_depth.
      rewrite (IH1 n Hn). reflexivity. }
    rewrite Hst, (IH2 (S n) Hw'). reflexivity.
Qed.

(* The fields after the first, made a pair. *)
Ltac tail_pair E :=
  match goal with
  | |- context [mk_unboxed_fields ?a ?b ?c ?d ?e ?g ?h ?i ?k] =>
    destruct (mk_unboxed_fields a b c d e g h i k) as [rest n'] eqn:E
  end;
  cbn [negb].

(* What the induction gives of [mk_unboxed_fields], made a pair. *)
Ltac from_ih IH next Hw E :=
  let Hu := fresh "Hu" in
  pose proof (IH next Hw) as Hu; rewrite E in Hu; exact Hu.

Lemma within_unboxes (n : nat) :
  ∀ (block : node) (fields : list (field * field_use)) (next : positive),
    calls_within F T cm dominated to_unbox n block fields ->
    unboxes F T cm dominated to_unbox block fields next
      (mk_unboxed_fields F T cm dominated to_unbox n block fields next).
Proof.
  induction n as [ | n IHn]; intros block fields;
    induction fields as [ | [f use] fields IHf]; intros next Hw;
    try (rewrite mk_nil; apply Unboxes_nil).
  all: pose proof (within_tail _ block f use fields Hw) as Hw'.
  all: pose proof (proj1 (within_unfold _ _ _) Hw) as Hw2.
  all: rewrite mk_cons; unfold step_with.
  all: destruct (is_data_field f) eqn:Hd; cbn [negb nest_depth].
  all: try solve [ tail_pair E;
                   apply Unboxes_left_out;
                   [ left; exact Hd | from_ih IHf next Hw' E ] ].
  all: destruct (get_single_field_source F T block f) as [ | source | ]
         eqn:Hs; cbn [negb nest_depth].
  all: try solve [ tail_pair E;
                   apply Unboxes_left_out;
                   [ right; exact Hs | from_ih IHf next Hw' E ] ].
  all: destruct use as [ | flow_to]; cbn [negb nest_depth].
  all: try solve [ tail_pair E;
                   apply Unboxes_unknown;
                   [ exact Hd | rewrite Hs; discriminate
                   | from_ih IHf (Pos.succ next) Hw' E ] ].
  all: destruct (forallb (has_to_be_unboxed dominated to_unbox) flow_to)
         eqn:Hall; cbn [negb nest_depth].
  all: try solve [ tail_pair E;
                   apply Unboxes_kept;
                   [ exact Hd | rewrite Hs; discriminate | exact Hall
                   | from_ih IHf (Pos.succ next) Hw' E ] ].
  all: try solve [ tail_pair E;
                   apply Unboxes_many;
                   [ exact Hd | exact Hs | exact Hall
                   | from_ih IHf (Pos.succ next) Hw' E ] ].
  all: pose proof (Hw2 source _
                     (Nested_call F T cm dominated to_unbox block
                        ((f, Use_known flow_to) :: fields) f flow_to source
                        (or_introl eq_refl) Hd Hs Hall))
         as Hn.
  - destruct Hn.
  - destruct (mk_unboxed_fields F T cm dominated to_unbox n source
                (fields_read F T cm flow_to) next) as [nested n1] eqn:E1.
    cbn [negb]. tail_pair E2.
    apply Unboxes_nested with (source := source) (next1 := n1);
      [ exact Hd | exact Hs | exact Hall
      | from_ih (IHn source (fields_read F T cm flow_to)) next Hn E1
      | from_ih IHf n1 Hw' E2 ].
Qed.

Lemma relation_ends (block : node) (fields : list (field * field_use))
    (next : positive) (r : unboxed_fields * positive) :
  unboxes F T cm dominated to_unbox block fields next r ->
  unboxing_ends F T cm dominated to_unbox block fields.
Proof.
  induction 1 as
      [ block next
      | block f use fields next r Hout H IH
      | block f fields next rest next' Hd Hs H IH
      | block f flow_to fields next rest next' Hd Hs Hall H IH
      | block f flow_to fields next rest next' Hd Hs Hall H IH
      | block f flow_to source fields next nested next1 rest next2 Hd Hs Hall
          H1 IH1 H2 IH2 ];
    constructor; intros s fs' Hc;
    destruct Hc as [f' flow_to' s' Hin Hd' Hs' Hall'];
    [ destruct Hin | | | | | ];
    destruct Hin as [Heq | Hin];
    try (destruct IH as [IH]; exact (IH _ _ (Nested_call F T cm dominated
                                               to_unbox block fields f'
                                               flow_to' s' Hin Hd' Hs'
                                               Hall')));
    try (destruct IH2 as [IH2]; exact (IH2 _ _ (Nested_call F T cm dominated
                                                  to_unbox block fields f'
                                                  flow_to' s' Hin Hd' Hs'
                                                  Hall')));
    inversion Heq; subst.
  - destruct Hout as [Hout | Hout]; congruence.
  - congruence.
  - congruence.
  - rewrite Hs' in Hs. injection Hs as <-. exact IH1.
Qed.

(* ======================================================================== *)
(* The bound                                                                *)
(* ======================================================================== *)

(* A chain of nested calls. *)
Inductive reach
    : node -> list (field * field_use) -> node -> list (field * field_use)
      -> Prop :=
  | Reach_refl (block : node) (fields : list (field * field_use)) :
      reach block fields block fields
  | Reach_step (block : node) (fields : list (field * field_use))
      (source : node) (fields' : list (field * field_use)) (block' : node)
      (fields'' : list (field * field_use)) :
      nested_call F T cm dominated to_unbox block fields source fields' ->
      reach source fields' block' fields'' ->
      reach block fields block' fields''.

Lemma reach_snoc (b : node) (fs : list (field * field_use)) (b' : node)
    (fs' : list (field * field_use)) (s : node)
    (fs'' : list (field * field_use)) :
  reach b fs b' fs' ->
  nested_call F T cm dominated to_unbox b' fs' s fs'' ->
  reach b fs s fs''.
Proof.
  induction 1 as [b fs | b fs s0 fs0 b1 fs1 Hc0 Hr IH]; intros Hc.
  - exact (Reach_step b fs s fs'' s fs'' Hc (Reach_refl s fs'')).
  - exact (Reach_step b fs s0 fs0 s fs'' Hc0 (IH Hc)).
Qed.

Lemma ends_no_cycle (block : node) (fields : list (field * field_use)) :
  unboxing_ends F T cm dominated to_unbox block fields ->
  ∀ (source : node) (fields' : list (field * field_use)),
    nested_call F T cm dominated to_unbox block fields source fields' ->
    reach source fields' block fields -> False.
Proof.
  induction 1 as [block fields Hends IH]. intros s fs' Hc Hr.
  inversion Hr as [b0 fs0 | b0 fs0 s2 fs2 b1 fs1 Hc2 Hr2]; subst.
  - exact (IH block fields Hc block fields Hc (Reach_refl block fields)).
  - exact (IH s fs' Hc s2 fs2 Hc2 (reach_snoc _ _ _ _ _ _ Hr2 Hc)).
Qed.

(* [all_masks n]: the lists of [n] booleans. *)
Fixpoint all_masks (n : nat) : list (list bool) :=
  match n with
  | O => [[]]
  | S n => map (cons true) (all_masks n) ++ map (cons false) (all_masks n)
  end.

Lemma all_masks_length (n : nat) : length (all_masks n) = 2 ^ n.
Proof.
  induction n as [ | n IH]; [ reflexivity | ].
  cbn [all_masks]. rewrite length_app, !length_map, IH, Nat.pow_succ_r'.
  lia.
Qed.

Lemma all_masks_in (m : list bool) : m ∈ all_masks (length m).
Proof.
  induction m as [ | b m IH]; [ left; reflexivity | ].
  cbn [all_masks length]. apply in_or_app.
  destruct b; [ left | right ]; apply in_map; exact IH.
Qed.

(* Which of the first names of the pairs of [t_usages T] are among
   [flow_to]; and the uses that it gives. *)
Definition mask (flow_to : list node) : list bool :=
  map (fun '(x, _) => mem_node x flow_to) (t_usages T).

Definition direct_of_mask (m : list bool) : list node :=
  Keyed.dedup node_eqb
    (flat_map
       (fun '(b, (x, y)) =>
          if b && mem_node y (t_has_usage T) then [y] else [])
       (combine m (t_usages T))).

Lemma direct_of_mask_spec (flow_to : list node) :
  get_direct_usages T flow_to = direct_of_mask (mask flow_to).
Proof.
  unfold get_direct_usages, direct_of_mask, mask. f_equal.
  induction (t_usages T) as [ | [x y] l IH]; [ reflexivity | ].
  cbn [map combine flat_map]. rewrite IH. reflexivity.
Qed.

Definition child (p : node * list bool) : node * list (field * field_use) :=
  (fst p,
   get_fields F T
     (add_usages_through_function_slots F T cm true (direct_of_mask (snd p)))).

(* Every call that the recursion may make. *)
Definition children : list (node * list (field * field_use)) :=
  map child
    (list_prod (map snd (t_sources T)) (all_masks (length (t_usages T)))).

Lemma children_length : length children = unboxing_depth T.
Proof.
  unfold children, unboxing_depth.
  rewrite length_map, length_prod, length_map, all_masks_length.
  reflexivity.
Qed.

Lemma source_in (block : node) (f : field) (source : node) :
  get_single_field_source F T block f = One_source source ->
  source ∈ map snd (t_sources T).
Proof.
  unfold get_single_field_source.
  destruct (existsb _ (f_constructor F)); [ discriminate | ].
  match goal with
  | |- (match ?l with _ => _ end) = _ -> _ =>
    destruct l as [ | s' [ | s'' rest]] eqn:Hd; try discriminate
  end.
  intros [= Heq]. subst s'.
  match type of Hd with
  | ?l = _ =>
    assert (Hin : source ∈ l) by (rewrite Hd; left; reflexivity)
  end.
  apply dedup_in in Hin.
  apply in_flat_map in Hin as ([[base f'] fsrc] & _ & Hin).
  destruct (node_eqb base block && field_eqb f f'); [ | destruct Hin ].
  apply in_flat_map in Hin as ([fs' src] & Hsrc & Hin).
  destruct (_ && _); [ | destruct Hin ].
  destruct Hin as [-> | []].
  apply in_map_iff. exists (fs', source).
  split; [ reflexivity | exact Hsrc ].
Qed.

Lemma call_in_children (block : node) (fields : list (field * field_use))
    (source : node) (fields' : list (field * field_use)) :
  nested_call F T cm dominated to_unbox block fields source fields' ->
  (source, fields') ∈ children.
Proof.
  intros Hc. destruct Hc as [f flow_to source _ _ Hs _].
  unfold children. apply in_map_iff. exists (source, mask flow_to). split.
  - unfold child, fields_read. cbn [fst snd]. rewrite direct_of_mask_spec.
    reflexivity.
  - apply in_prod; [ exact (source_in _ _ _ Hs) | ].
    replace (length (t_usages T)) with (length (mask flow_to))
      by (unfold mask; apply length_map).
    apply all_masks_in.
Qed.

(* The calls [visited] on the chain down to [block] and [fields], which
   all reach it, are distinct and among [children]: what is left of
   [children] bounds the chains from there. *)
Lemma within_from (n : nat) :
  ∀ (block : node) (fields : list (field * field_use))
    (visited : list (node * list (field * field_use))),
    unboxing_ends F T cm dominated to_unbox block fields ->
    NoDup visited -> incl visited children ->
    (∀ y, y ∈ visited -> reach (fst y) (snd y) block fields) ->
    length visited + n = length children ->
    calls_within F T cm dominated to_unbox n block fields.
Proof.
  induction n as [ | n IH]; intros block fields visited He Hnd Hinc Hreach
    Hlen; apply within_unfold; intros s fs' Hc.
  all: assert (Hnot : ¬ (s, fs') ∈ visited)
         by (intros Hin; exact (ends_no_cycle block fields He s fs' Hc
                                  (Hreach _ Hin))).
  all: assert (Hinc' : incl ((s, fs') :: visited) children)
         by (intros y [<- | Hy];
             [ exact (call_in_children _ _ _ _ Hc) | exact (Hinc y Hy) ]).
  all: assert (Hle : length ((s, fs') :: visited) <= length children)
         by (apply NoDup_incl_length; [ constructor; assumption
                                      | exact Hinc' ]).
  - cbn [length] in Hle. lia.
  - apply (IH s fs' ((s, fs') :: visited)).
    + destruct He as [He]. exact (He s fs' Hc).
    + constructor; assumption.
    + exact Hinc'.
    + intros y [<- | Hy]; [ apply Reach_refl | ].
      exact (reach_snoc _ _ _ _ _ _ (Hreach y Hy) Hc).
    + cbn [length]. lia.
Qed.

Lemma ends_within_section (block : node) (fields : list (field * field_use)) :
  unboxing_ends F T cm dominated to_unbox block fields ->
  calls_within F T cm dominated to_unbox (unboxing_depth T) block fields.
Proof.
  intros He.
  apply (within_from _ block fields []);
    [ exact He | constructor | intros y [] | intros y []
    | rewrite children_length; reflexivity ].
Qed.

Lemma within_ends_section (n : nat) :
  ∀ (block : node) (fields : list (field * field_use)),
    calls_within F T cm dominated to_unbox n block fields ->
    unboxing_ends F T cm dominated to_unbox block fields.
Proof.
  induction n as [ | n IH]; intros block fields Hw; constructor;
    intros s fs' Hc; pose proof (proj1 (within_unfold _ _ _) Hw s fs' Hc)
      as Hw';
    [ destruct Hw' | exact (IH s fs' Hw') ].
Qed.


(* ======================================================================== *)
(* Where the recursion does not end                                         *)
(* ======================================================================== *)

(* The name [k] after [p]. *)
Fixpoint succ_n (k : nat) (p : positive) : positive :=
  match k with
  | O => p
  | S k => succ_n k (Pos.succ p)
  end.

Lemma succ_n_add (a b : nat) (p : positive) :
  succ_n (a + b) p = succ_n b (succ_n a p).
Proof.
  revert p. induction a as [ | a IH]; intros p; [ reflexivity | ].
  cbn [Nat.add succ_n]. apply IH.
Qed.

(* The number of variables of an unboxing. *)
Definition leaf_count (fields : unboxed_fields) : nat :=
  length (fields_leaves fields).

Lemma leaf_count_var (f : field) (v : variable) (rest : unboxed_fields) :
  leaf_count ((f, Not_unboxed v) :: rest) = S (leaf_count rest).
Proof. reflexivity. Qed.

Lemma leaf_count_nested (f : field) (nested rest : unboxed_fields) :
  leaf_count ((f, Unboxed nested) :: rest)
  = leaf_count nested + leaf_count rest.
Proof.
  unfold leaf_count. cbn [fields_leaves unboxed_field_leaves].
  rewrite length_app. reflexivity.
Qed.

(* What the induction gives of the name after the variables. *)
Ltac from_ih_next IH next E :=
  let Hu := fresh "Hu" in
  pose proof (IH next) as Hu; rewrite E in Hu; cbn [fst snd] in Hu |- *;
  exact Hu.

(* The name after the last variable is given by their number. *)
Lemma mk_next (n : nat) :
  ∀ (block : node) (fields : list (field * field_use)) (next : positive),
    snd (mk_unboxed_fields F T cm dominated to_unbox n block fields next)
    = succ_n
        (leaf_count
           (fst (mk_unboxed_fields F T cm dominated to_unbox n block fields
                   next)))
        next.
Proof.
  induction n as [ | n IHn]; intros block fields;
    induction fields as [ | [f use] fields IHf]; intros next;
    try (rewrite mk_nil; reflexivity).
  all: rewrite mk_cons; unfold step_with.
  all: destruct (is_data_field f); cbn [negb nest_depth];
    [ | tail_pair E; from_ih_next IHf next E ].
  all: destruct (get_single_field_source F T block f) as [ | source | ];
    cbn [negb nest_depth];
    [ tail_pair E; from_ih_next IHf next E | | ].
  all: destruct use as [ | flow_to]; cbn [negb nest_depth];
    [ tail_pair E; cbn [fst snd]; rewrite leaf_count_var;
      cbn [succ_n]; from_ih_next IHf (Pos.succ next) E | ].
  all: destruct (forallb (has_to_be_unboxed dominated to_unbox) flow_to);
    cbn [negb nest_depth];
    try (tail_pair E; cbn [fst snd]; rewrite leaf_count_var;
         cbn [succ_n]; from_ih_next IHf (Pos.succ next) E).
  destruct (mk_unboxed_fields F T cm dominated to_unbox n source
              (fields_read F T cm flow_to) next) as [nested n1] eqn:E1.
  cbn [negb]. tail_pair E2. cbn [fst snd].
  rewrite leaf_count_nested, succ_n_add.
  pose proof (IHn source (fields_read F T cm flow_to) next) as H1.
  rewrite E1 in H1. cbn [fst snd] in H1. rewrite <- H1.
  pose proof (IHf n1) as H2. rewrite E2 in H2. cbn [fst snd] in H2.
  exact H2.
Qed.

(* The variables of a field, and whether it is kept. *)
Definition opt_count (u : option unboxed_field) : nat :=
  match u with
  | None => 0
  | Some (Not_unboxed _) => 1
  | Some (Unboxed fields) => leaf_count fields
  end.

Definition present {A : Type} (u : option A) : bool :=
  match u with
  | Some _ => true
  | None => false
  end.

Lemma step_present (nest1 nest2 : node -> list (field * field_use)
                                   -> positive
                                   -> option (unboxed_fields * positive))
    (block : node) (f : field) (use : field_use) (next : positive) :
  present (fst (step_with nest1 block f use next))
  = present (fst (step_with nest2 block f use next)).
Proof.
  unfold step_with.
  destruct (is_data_field f); [ | reflexivity ]. cbn [negb].
  destruct (get_single_field_source F T block f); try reflexivity.
  destruct use as [ | flow_to]; [ reflexivity | ].
  destruct (forallb (has_to_be_unboxed dominated to_unbox) flow_to);
    [ | reflexivity ].
  destruct (nest1 _ _ next) as [[] | ]; destruct (nest2 _ _ next) as [[] | ];
    reflexivity.
Qed.

Lemma step_next (n : nat) (block : node) (f : field) (use : field_use)
    (next : positive) :
  snd (step_with (nest_depth n) block f use next)
  = succ_n (opt_count (fst (step_with (nest_depth n) block f use next)))
      next.
Proof.
  unfold step_with.
  destruct (is_data_field f); [ | reflexivity ]. cbn [negb].
  destruct (get_single_field_source F T block f) as [ | source | ];
    try reflexivity;
    destruct use as [ | flow_to]; try reflexivity;
    destruct (forallb (has_to_be_unboxed dominated to_unbox) flow_to);
    try reflexivity.
  destruct n as [ | n]; [ reflexivity | ]. unfold nest_depth.
  pose proof (mk_next n source (fields_read F T cm flow_to) next) as H.
  destruct (mk_unboxed_fields F T cm dominated to_unbox n source
              (fields_read F T cm flow_to) next) as [nested n1].
  cbn [fst snd opt_count] in H |- *. exact H.
Qed.

Lemma step_head_call (n : nat) (block : node) (f : field)
    (flow_to : list node) (source : node) (next : positive) :
  is_data_field f = true ->
  get_single_field_source F T block f = One_source source ->
  forallb (has_to_be_unboxed dominated to_unbox) flow_to = true ->
  fst (step_with (nest_depth n) block f (Use_known flow_to) next)
  = Some (match n with
          | O => Not_unboxed next
          | S n =>
            Unboxed
              (fst (mk_unboxed_fields F T cm dominated to_unbox n source
                      (fields_read F T cm flow_to) next))
          end).
Proof.
  intros Hd Hs Hall. unfold step_with. rewrite Hd, Hs, Hall.
  cbn [negb]. destruct n as [ | n]; [ reflexivity | ]. unfold nest_depth.
  destruct (mk_unboxed_fields F T cm dominated to_unbox n source
              (fields_read F T cm flow_to) next).
  reflexivity.
Qed.

(* The head of the fields of two depths, which agree. *)
Ltac agree_head m Heq IHf Hc :=
  rewrite !mk_cons in Heq;
  lazymatch type of Hc with
  | nested_call _ _ _ _ _ ?block ((?f, ?use) :: ?fields) _ _ =>
    lazymatch type of Heq with
    | fst (let '(_, _) := step_with _ _ _ _ ?next in _) = _ =>
      pose proof (step_present (nest_depth m) (nest_depth (S m)) block f
                    use next) as Hsome;
      pose proof (step_next m block f use next) as Hx1;
      pose proof (step_next (S m) block f use next) as Hx2;
      destruct (step_with (nest_depth m) block f use next)
        as [u1 x1] eqn:Hs1;
      destruct (step_with (nest_depth (S m)) block f use next)
        as [u2 x2] eqn:Hs2;
      cbn [fst snd] in Hsome, Hx1, Hx2;
      destruct (mk_unboxed_fields F T cm dominated to_unbox m block fields x1)
        as [r1 y1] eqn:Hr1;
      destruct (mk_unboxed_fields F T cm dominated to_unbox (S m) block fields
                  x2) as [r2 y2] eqn:Hr2;
      cbn [fst] in Heq;
      assert (Hu : u1 = u2 ∧ r1 = r2)
        by (destruct u1, u2; cbn [present] in Hsome; try discriminate;
            [ injection Heq as -> ->; split; reflexivity
            | split; [ reflexivity | exact Heq ] ]);
      destruct Hu as [<- <-];
      let Hx := fresh "Hx" in
      assert (Hx : x2 = x1) by (rewrite Hx1, Hx2; reflexivity);
      rewrite Hx in Hr2;
      assert (Htail : calls_within F T cm dominated to_unbox m block fields)
        by (apply (IHf x1); rewrite Hr1, Hr2; reflexivity);
      let f' := fresh "f'" in
      let ft := fresh "ft" in
      let s' := fresh "s'" in
      let Hin := fresh "Hin" in
      let Hd := fresh "Hd" in
      let Hsrc := fresh "Hsrc" in
      let Hall := fresh "Hall" in
      let Hh := fresh "Hh" in
      destruct Hc as [f' ft s' Hin Hd Hsrc Hall];
      destruct Hin as [Hh | Hin];
      [ injection Hh as -> ->;
        pose proof (step_head_call m block f' ft s' next Hd Hsrc Hall) as H1;
        pose proof (step_head_call (S m) block f' ft s' next Hd Hsrc Hall)
          as H2;
        rewrite Hs1 in H1; rewrite Hs2 in H2; cbn [fst] in H1, H2;
        rewrite H1 in H2
      | exact (proj1 (within_unfold _ _ _)
                 Htail s' _ (Nested_call F T cm dominated to_unbox block
                               fields f' ft s' Hin Hd Hsrc Hall)) ]
    end
  end.

(* Two depths that agree bound the chains by the first. *)
Lemma agree_within (n : nat) :
  ∀ (block : node) (fields : list (field * field_use)) (next : positive),
    fst (mk_unboxed_fields F T cm dominated to_unbox n block fields next)
    = fst (mk_unboxed_fields F T cm dominated to_unbox (S n) block fields
             next) ->
    calls_within F T cm dominated to_unbox n block fields.
Proof.
  induction n as [ | n IHn]; intros block fields;
    induction fields as [ | [f use] fields IHf]; intros next Heq;
    apply (proj2 (within_unfold _ _ _));
    intros s fs' Hc;
    try (destruct Hc as [f' ft s' Hin]; destruct Hin; fail).
  - agree_head 0 Heq IHf Hc. discriminate H2.
  - agree_head (S n) Heq IHf Hc. injection H2 as H2. exact (IHn _ _ _ H2).
Qed.

End Fuel.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem full_depth : full_depth_statement.
Proof. exact mk_unboxed_fields_full_depth. Qed.

Theorem ends_within : ends_within_statement.
Proof. exact ends_within_section. Qed.

Theorem within_ends : within_ends_statement.
Proof. exact within_ends_section. Qed.

Theorem unboxes_ends : unboxes_ends_statement.
Proof.
  intros F T cm dominated to_unbox block fields. split.
  - intros He next.
    exists (mk_unboxed_fields F T cm dominated to_unbox (unboxing_depth T)
              block fields next).
    apply within_unboxes. apply ends_within_section. exact He.
  - intros next r H. exact (relation_ends F T cm dominated to_unbox _ _ _ _ H).
Qed.

Theorem fuel_enough : fuel_enough_statement.
Proof.
  intros F T cm dominated to_unbox block fields next r H.
  rewrite mk_unboxed_fields_full_depth.
  apply (unboxes_mk F T cm dominated to_unbox _ _ _ _ H).
  apply ends_within_section.
  exact (relation_ends F T cm dominated to_unbox _ _ _ _ H).
Qed.

Theorem more_fuel : more_fuel_statement.
Proof.
  intros F T cm dominated to_unbox block fields next n He Hle.
  rewrite mk_unboxed_fields_full_depth.
  apply (within_stable F T cm dominated to_unbox (unboxing_depth T));
    [ apply ends_within_section; exact He | exact Hle ].
Qed.

Theorem fuel_changes : fuel_changes_statement.
Proof.
  intros F T cm dominated to_unbox block fields next n Hne Heq.
  apply Hne. apply (within_ends_section F T cm dominated to_unbox n).
  exact (agree_within F T cm dominated to_unbox n block fields next Heq).
Qed.

Theorem ends_stable : ends_stable_statement.
Proof.
  intros F T cm dominated to_unbox block fields next. split.
  - intros He.
    rewrite (more_fuel F T cm dominated to_unbox block fields next
               (unboxing_depth T) He (le_n _)),
      (more_fuel F T cm dominated to_unbox block fields next
         (S (unboxing_depth T)) He (le_S _ _ (le_n _))).
    reflexivity.
  - intros Heq.
    apply (within_ends_section F T cm dominated to_unbox (unboxing_depth T)).
    exact (agree_within F T cm dominated to_unbox _ block fields next Heq).
Qed.

Theorem name_unboxed_fields_relation :
  name_unboxed_fields_relation_statement.
Proof.
  intros F T cm dominated to_unbox a x next r H.
  unfold name_unboxed_fields. exact (fuel_enough _ _ _ _ _ _ _ _ _ H).
Qed.

(* ======================================================================== *)
(* The model's unboxings at that depth                                      *)
(* ======================================================================== *)

Lemma name_unboxed_fields_depth (F : facts) (T : tables)
    (cm : finite_map variable) (dominated : list (node * node))
    (to_unbox : list node) (a x : node) (next : positive) :
  name_unboxed_fields F T cm dominated to_unbox a x next
  = mk_unboxed_fields F T cm dominated to_unbox (unboxing_depth T) a
      (get_fields F T
         (add_usages_through_function_slots F T cm true
            (get_direct_usages T [x])))
      next.
Proof. apply mk_unboxed_fields_full_depth. Qed.

Lemma fold_left_ext_pointwise {A B : Type} (f g : A -> B -> A) (l : list B) :
  (∀ (a : A) (b : B), f a b = g a b) ->
  ∀ a : A, List.fold_left f l a = List.fold_left g l a.
Proof.
  intros Hfg. induction l as [ | b l IH]; intros a; [ reflexivity | ].
  cbn [List.fold_left]. rewrite Hfg. apply IH.
Qed.

(* [changed_representations] with [mk_unboxed_fields] at the depth
   [unboxing_depth T]. *)
Lemma changed_representations_depth (F : facts) (T : tables)
    (cm : finite_map variable) (dominated : list (node * node))
    (to_unbox : list node) (is_local : field -> bool) (tcr : list node)
    (next : positive) :
  changed_representations F T cm dominated to_unbox is_local tcr next
  = let '(acc, _, next) :=
      List.fold_left
        (fun '(acc, seen, next) x =>
           if mem_node x seen
           then (acc, seen, next)
           else
             match set_function_slots F x with
             | [] => (acc, seen, next)
             | l =>
               let '(fields, next) :=
                 mk_unboxed_fields F T cm dominated to_unbox
                   (unboxing_depth T) x
                   (get_fields_of_constructors F T is_local (map snd l)) next
               in
               let '(fss, next) := rename_slots (map fst l) next in
               (acc
                ++ flat_map
                     (fun '(fs, f) =>
                        flat_map
                          (fun '(a, c) =>
                             if node_eqb a f
                             then [(c, Closure_repr fields fss fs)]
                             else [])
                          dominated)
                     l,
                seen ++ map snd l, next)
             end)
        tcr ([], [], next)
    in
    (acc, next).
Proof.
  unfold changed_representations.
  erewrite fold_left_ext_pointwise; [ reflexivity | ].
  intros [[acc seen] n] x. cbv beta iota.
  destruct (mem_node x seen); [ reflexivity | ].
  destruct (set_function_slots F x) as [ | p l]; [ reflexivity | ].
  rewrite mk_unboxed_fields_full_depth. reflexivity.
Qed.

(* ======================================================================== *)
(* A recursion that does not end                                            *)
(* ======================================================================== *)

Module LF := LoopField.

Lemma loop_fields :
  LF.fields_10
  = [(Field_block 0 Value, Use_known [Node_variable 11%positive]);
     (Field_block 1 Value, Use_unknown)].
Proof. vm_compute. reflexivity. Qed.

Lemma loop_fields_11 :
  fields_read LF.facts LF.tables LF.my_closures [Node_variable 11%positive]
  = LF.fields_10.
Proof. vm_compute. reflexivity. Qed.

Lemma loop_source_0 :
  get_single_field_source LF.facts LF.tables (Node_variable 10%positive)
    (Field_block 0 Value)
  = One_source (Node_variable 10%positive).
Proof. vm_compute. reflexivity. Qed.

Lemma loop_source_1 :
  get_single_field_source LF.facts LF.tables (Node_variable 10%positive)
    (Field_block 1 Value)
  <> No_source.
Proof. vm_compute. discriminate. Qed.

Lemma loop_to_unbox_11 :
  forallb (has_to_be_unboxed LF.dominated LF.to_unbox)
    [Node_variable 11%positive]
  = true.
Proof. vm_compute. reflexivity. Qed.

(* The depth [n] on [x10]: field 0 nested [n] levels, field 1 a
   variable. *)
Lemma loop_zero (next : positive) :
  mk_unboxed_fields LF.facts LF.tables LF.my_closures LF.dominated
    LF.to_unbox 0 (Node_variable 10%positive) LF.fields_10 next
  = ([(Field_block 0 Value, Not_unboxed next);
      (Field_block 1 Value, Not_unboxed (Pos.succ next))],
     Pos.succ (Pos.succ next)).
Proof.
  rewrite loop_fields, mk_cons; unfold step_with;
    cbn [is_data_field negb nest_depth];
    rewrite loop_source_0, loop_to_unbox_11; cbn [nest_depth].
  rewrite mk_cons. unfold step_with. cbn [is_data_field negb].
  destruct (get_single_field_source LF.facts LF.tables
              (Node_variable 10%positive) (Field_block 1 Value))
    eqn:Hs1; [ exfalso; exact (loop_source_1 Hs1) | | ];
    rewrite mk_nil; reflexivity.
Qed.

Lemma loop_succ (n : nat) (next : positive) :
  mk_unboxed_fields LF.facts LF.tables LF.my_closures LF.dominated
    LF.to_unbox (S n) (Node_variable 10%positive) LF.fields_10 next
  = let '(nested, n1) :=
      mk_unboxed_fields LF.facts LF.tables LF.my_closures LF.dominated
        LF.to_unbox n (Node_variable 10%positive) LF.fields_10 next
    in
    ([(Field_block 0 Value, Unboxed nested);
      (Field_block 1 Value, Not_unboxed n1)],
     Pos.succ n1).
Proof.
  rewrite loop_fields at 1. rewrite mk_cons; unfold step_with;
    cbn [is_data_field negb nest_depth];
    rewrite loop_source_0, loop_to_unbox_11; cbn [nest_depth].
  rewrite loop_fields_11.
  destruct (mk_unboxed_fields LF.facts LF.tables LF.my_closures
              LF.dominated LF.to_unbox n (Node_variable 10%positive)
              LF.fields_10 next) as [nested n1].
  cbn [negb]. rewrite mk_cons. unfold step_with. cbn [is_data_field negb].
  destruct (get_single_field_source LF.facts LF.tables
              (Node_variable 10%positive) (Field_block 1 Value))
    eqn:Hs1; [ exfalso; exact (loop_source_1 Hs1) | | ];
    rewrite mk_nil; reflexivity.
Qed.

Lemma loop_nesting (n : nat) :
  ∀ next : positive,
    nesting
      (fst (mk_unboxed_fields LF.facts LF.tables LF.my_closures LF.dominated
              LF.to_unbox n (Node_variable 10%positive) LF.fields_10 next))
    = n.
Proof.
  induction n as [ | n IH]; intros next;
    [ rewrite loop_zero; reflexivity | ].
  rewrite loop_succ. specialize (IH next).
  destruct (mk_unboxed_fields LF.facts LF.tables LF.my_closures
              LF.dominated LF.to_unbox n (Node_variable 10%positive)
              LF.fields_10 next) as [nested n1].
  cbn [fst] in IH |- *. unfold nesting in IH |- *.
  cbn [unboxed_nesting Nat.max Nat.pred] in IH |- *. lia.
Qed.

Lemma pairs_head_none (f : field) (u u' : unboxed_field)
    (rest rest' : unboxed_fields) :
  unboxed_field_pairs (field_kind f) u u' = None ->
  unboxed_field_pairs Value (Unboxed ((f, u) :: rest))
    (Unboxed ((f, u') :: rest'))
  = None.
Proof.
  intros H. cbn [unboxed_field_pairs Keyed.lookup].
  rewrite field_eqb_refl, H. reflexivity.
Qed.

Lemma loop_not_alike (n : nat) :
  ∀ next next' : positive,
    unboxed_field_pairs Value
      (Unboxed
         (fst (mk_unboxed_fields LF.facts LF.tables LF.my_closures
                 LF.dominated LF.to_unbox n (Node_variable 10%positive)
                 LF.fields_10 next)))
      (Unboxed
         (fst (mk_unboxed_fields LF.facts LF.tables LF.my_closures
                 LF.dominated LF.to_unbox (S n) (Node_variable 10%positive)
                 LF.fields_10 next')))
    = None.
Proof.
  induction n as [ | n IH]; intros next next';
    [ rewrite loop_succ, loop_zero; reflexivity | ].
  rewrite (loop_succ n next), (loop_succ (S n) next').
  specialize (IH next next').
  destruct (mk_unboxed_fields LF.facts LF.tables LF.my_closures
              LF.dominated LF.to_unbox n (Node_variable 10%positive)
              LF.fields_10 next) as [nested n1].
  destruct (mk_unboxed_fields LF.facts LF.tables LF.my_closures
              LF.dominated LF.to_unbox (S n) (Node_variable 10%positive)
              LF.fields_10 next') as [nested' n1'].
  cbn [fst] in IH |- *. apply pairs_head_none. exact IH.
Qed.

Theorem loop_field : loop_field_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { intros He.
    apply (ends_no_cycle LF.facts LF.tables LF.my_closures LF.dominated
             LF.to_unbox _ _ He (Node_variable 10%positive) LF.fields_10).
    - rewrite <- loop_fields_11 at 2.
      apply (Nested_call _ _ _ _ _ _ _ (Field_block 0 Value)).
      + rewrite loop_fields. left. reflexivity.
      + reflexivity.
      + exact loop_source_0.
      + exact loop_to_unbox_11.
    - apply Reach_refl. }
  split; [ exact loop_nesting | ].
  split; [ exact loop_not_alike | ].
  split; vm_compute; reflexivity.
Qed.
