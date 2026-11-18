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

(* The proofs of FlambdaReaperWrapperSpec.v. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaSemanticsSpec
  FlambdaValueSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperUnboxingSpec
  FlambdaReaperWrapperSpec.
From OxCamlFlambda2 Require FlambdaMapProofs FlambdaReaperFlowProofs
  FlambdaReaperLeastProofs FlambdaReaperUnboxingProofs
  FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The arm reached                                                          *)
(* ======================================================================== *)

Theorem no_source_arm : no_source_arm_statement.
Proof.
  hnf. split.
  { apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
    vm_compute. reflexivity. }
  repeat match goal with |- _ ∧ _ => split end;
  lazymatch goal with
  | |- behaves _ _ =>
      exists 40%nat; exists FlambdaValueSpec.no_reads; vm_compute;
      reflexivity
  | |- _ => vm_compute; reflexivity
  end.
Qed.

(* ======================================================================== *)
(* What the least solution carries along aliases                            *)
(* ======================================================================== *)

Section Least.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Hypothesis Hleast : least_solution is_local F T.

Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).

Lemma closedS : closed is_local G S.
Proof. exact (FlambdaReaperLeastProofs.least_closed is_local F T Hleast). Qed.

Lemma holdsS (a : atom) : derivable is_local G S a -> holds S a = true.
Proof. exact (FlambdaReaperLeastProofs.least_holds is_local F T Hleast a). Qed.

Lemma derivS (a : atom) : holds S a = true -> derivable is_local G S a.
Proof.
  exact (FlambdaReaperLeastProofs.least_derivable is_local F T Hleast a).
Qed.

Lemma no_stale_usages (x u : node) :
  s_usages S x u = true -> s_any_usage S x = false.
Proof.
  exact (proj1 (FlambdaReaperLeastProofs.least_no_stale is_local F T x u
                  Hleast)).
Qed.

Lemma no_stale_sources (x s : node) :
  s_sources S x s = true -> s_any_source S x = false.
Proof.
  exact (proj2 (FlambdaReaperLeastProofs.least_no_stale is_local F T x s
                  Hleast)).
Qed.

(* [path x y]: what is at [x] is at [y] by aliases. *)
Inductive path : node -> node -> Prop :=
  | path_refl : ∀ x, path x x
  | path_step : ∀ x y z, s_alias S y x = true -> path y z -> path x z.

Lemma path_trans (x y z : node) : path x y -> path y z -> path x z.
Proof.
  intros H1 H2. induction H1 as [ | x y' y Ha _ IH]; [ exact H2 | ].
  exact (path_step _ _ _ Ha (IH H2)).
Qed.

Lemma path_snoc (x y z : node) :
  path x y -> s_alias S z y = true -> path x z.
Proof.
  intros H Ha. apply (path_trans _ _ _ H). apply (path_step _ z);
    [ exact Ha | apply path_refl ].
Qed.

Lemma path_one (x y : node) : s_alias S y x = true -> path x y.
Proof. intros Ha. exact (path_step _ _ _ Ha (path_refl y)). Qed.

(* An allocation of the sources of a node reaches it by aliases. *)
Lemma sources_path (n s : node) : s_sources S n s = true -> path s n.
Proof.
  intros H. pose proof (derivS (Atom_sources n s) H) as D.
  remember (Atom_sources n s) as at0 eqn:E. revert n s E H.
  induction D; intros n0 s0 E Hn; try discriminate E; injection E as <- <-.
  - apply path_refl.
  - apply path_refl.
  - apply (path_snoc _ from); [ | exact (holdsS _ D2) ].
    apply (IHD1 from source eq_refl). exact (holdsS _ D1).
Qed.

Lemma any_usage_back (x y : node) :
  path x y -> s_any_usage S y = true -> s_any_usage S x = true.
Proof.
  intros H. induction H as [ | x y' z Ha _ IH]; intros Hu; [ exact Hu | ].
  exact (any_usage_alias _ _ _ closedS y' x (IH Hu) Ha).
Qed.

Lemma any_source_fwd (x y : node) :
  path x y -> s_any_source S x = true -> s_any_source S y = true.
Proof.
  intros H. induction H as [ | x y' z Ha _ IH]; intros Hs; [ exact Hs | ].
  exact (IH (any_source_alias _ _ _ closedS x y' Ha Hs)).
Qed.

Lemma sources_fwd (x y s : node) :
  path x y -> s_sources S x s = true ->
  s_any_source S y = true ∨ s_sources S y s = true.
Proof.
  intros H. induction H as [ | x y' z Ha Hp IH]; intros Hs;
    [ right; exact Hs | ].
  destruct (s_any_source S y') eqn:Ey'.
  - left. exact (any_source_fwd _ _ Hp Ey').
  - apply IH. exact (sources_alias _ _ _ closedS x y' s Hs
                       (no_stale_sources _ _ Hs) Ha Ey').
Qed.

Lemma usages_back (x y u : node) :
  path x y -> s_usages S y u = true ->
  s_any_usage S x = true ∨ s_usages S x u = true.
Proof.
  intros H. induction H as [ | x y' z Ha _ IH]; intros Hu;
    [ right; exact Hu | ].
  destruct (IH Hu) as [Hy' | Hy'].
  - left. exact (any_usage_alias _ _ _ closedS y' x Hy' Ha).
  - destruct (s_any_usage S x) eqn:Ex; [ left; reflexivity | right ].
    exact (usages_alias _ _ _ closedS y' x u Hy' (no_stale_usages _ _ Hy')
             Ha Ex).
Qed.

(* A use of a node ends, by aliases, in any usage, a read with a used
   result, or a node given an argument with a source. *)
Lemma has_usage_end (x : node) :
  s_has_usage S x = true ->
  ∃ y, path x y
       ∧ (s_any_usage S y = true
          ∨ (∃ v f, g_accessor G v f y ∧ s_has_usage S v = true)
          ∨ (∃ from r, g_argument G from r y ∧ s_has_source S from = true)).
Proof.
  intros H. pose proof (derivS (Atom_has_usage x) H) as D.
  remember (Atom_has_usage x) as at0 eqn:E. revert x E H.
  induction D; intros x0 E Hx; try discriminate E; injection E as <-.
  - exists x. split; [ apply path_refl | left; exact (holdsS _ D) ].
  - destruct (IHD1 to_ eq_refl (holdsS _ D1)) as (y & Hp & Hy).
    exists y. split; [ | exact Hy ].
    exact (path_step _ _ _ (holdsS _ D2) Hp).
  - exists base. split; [ apply path_refl | right; left ].
    exists to_, relation. split; [ exact H | exact (holdsS _ D) ].
  - exists base. split; [ apply path_refl | right; right ].
    exists from, relation. split; [ exact H | exact (holdsS _ D) ].
Qed.

(* A node with a source and not any source has an allocation with a
   source among its sources. *)
Lemma has_source_sources (x : node) :
  s_has_source S x = true -> s_any_source S x = false ->
  ∃ s, s_sources S x s = true ∧ s_has_source S s = true.
Proof.
  intros H. pose proof (derivS (Atom_has_source x) H) as D.
  remember (Atom_has_source x) as at0 eqn:E. revert x E H.
  induction D; intros x0 E Hx Hany; try discriminate E; injection E as <-.
  - pose proof (holdsS _ D) as Ha. cbn [holds] in Ha. rewrite Ha in Hany.
    discriminate.
  - pose proof (holdsS _ D1) as Hf. pose proof (holdsS _ D2) as Ha.
    cbn [holds] in Hf, Ha.
    destruct (s_any_source S from) eqn:Ef.
    + rewrite (any_source_alias _ _ _ closedS from to_ Ha Ef) in Hany.
      discriminate.
    + destruct (IHD1 from eq_refl Hf Ef) as (s & Hs & Hhs).
      exists s. split; [ | exact Hhs ].
      exact (sources_alias _ _ _ closedS from to_ s Hs Ef Ha Hany).
  - exists base. split; [ | exact Hx ].
    exact (sources_constructor _ _ _ closedS from relation base H Hany).
  - exists base. split; [ | exact Hx ].
    exact (sources_constructor _ _ _ closedS from relation base H Hany).
  - exists base. split; [ | exact Hx ].
    exact (sources_parameter _ _ _ closedS to_ relation base H Hany).
Qed.

(* ======================================================================== *)
(* A result and the parameter it is given to                                *)
(* ======================================================================== *)

Variable cannot_change : list node.
Variable max_unbox_size : nat.
Variable lambda_lifting : bool.
Variable dominated : list (node * node).
Variable to_unbox : list node.
Hypothesis Hdom : check_dominated T dominated = true.
Hypothesis Hunbox :
  check_to_unbox_fixed is_local F T cannot_change max_unbox_size
    lambda_lifting to_unbox
  = true.

Lemma dom_iff (a x : node) :
  mem_pair a x dominated = true <-> dominated_by_allocation_point S a x.
Proof.
  exact (FlambdaReaperUnboxingProofs.check_dominated_sound T dominated Hdom
           a x).
Qed.

(* An allocation to unbox is not in [cannot_unbox0]. *)
Lemma unboxed_rules :
  ∃ (crr0 crr1 crr cu0 cu : node -> bool),
    unboxing_closed is_local G S (fun n => mem_node n cannot_change)
      max_unbox_size lambda_lifting crr0 crr1 crr cu0 cu
    ∧ fix_closed G S cu0
    ∧ ∀ a, mem_node a to_unbox = true -> cu0 a = false.
Proof.
  destruct (FlambdaReaperUnboxingProofs.check_to_unbox_fixed_sound is_local F
              T cannot_change max_unbox_size lambda_lifting to_unbox Hunbox)
    as (crr0 & crr1 & crr & cu0 & cu & Huc & Hfix & Hto & _).
  exists crr0, crr1, crr, cu0, cu.
  split; [ exact Huc | split; [ exact Hfix | ] ].
  intros a Ha. destruct (Hto a Ha) as [_ Hcu].
  destruct (cu0 a) eqn:E; [ | reflexivity ].
  rewrite (cu_cu0 _ _ _ _ _ _ _ _ _ _ _ Huc a E) in Hcu. discriminate.
Qed.

(* [r] given to [x]: an [x] with a use takes from [r] the allocation to
   unbox that dominates [r], and no other. *)
Lemma flow_kept (x r a : node) :
  check_argument_reads F = true ->
  s_alias S x r = true -> mem_pair a r dominated = true ->
  mem_node a to_unbox = true -> s_has_usage S x = true ->
  mem_pair a x dominated = true.
Proof.
  intros Hargs Hxr Hr Ha Hx.
  destruct unboxed_rules as (crr0 & crr1 & crr & cu0 & cu & Huc & Hfix & Hto).
  pose proof (Hto a Ha) as Hcu0.
  assert (Hau : s_any_usage S a = false).
  { destruct (s_any_usage S a) eqn:E; [ | reflexivity ].
    rewrite (cu0_any_usage _ _ _ _ _ _ _ _ _ _ _ Huc a E) in Hcu0.
    discriminate. }
  apply dom_iff in Hr. destruct Hr as (Hsra & Hhsa & _).
  pose proof (path_snoc _ _ _
                (sources_path _ _ Hsra) Hxr) as Hpax.
  destruct (has_usage_end x Hx) as (y & Hpxy & Hend).
  pose proof (path_trans _ _ _ Hpax Hpxy) as Hpay.
  assert (Hauy : s_any_usage S y = false).
  { destruct (s_any_usage S y) eqn:E; [ | reflexivity ].
    rewrite (any_usage_back _ _ Hpay E) in Hau. discriminate. }
  assert (Hread : ∃ v f, g_accessor G v f y ∧ s_has_usage S v = true).
  { destruct Hend as [Hy | [Hy | (from & r' & Harg & _)]];
      [ rewrite Hy in Hauy; discriminate | exact Hy | ].
    unfold check_argument_reads in Hargs. rewrite forallb_forall in Hargs.
    specialize (Hargs _ Harg). cbn beta iota in Hargs.
    apply existsb_exists in Hargs as ([[v f] y'] & Hin & Ht).
    apply andb_true_iff in Ht as [Hy' Hv]. apply node_eqb_eq in Hy'.
    subst y'. exists v, f. split; [ exact Hin | ].
    apply (has_usage_any _ _ _ closedS).
    apply (any_usage_input _ _ _ closedS).
    apply mem_node_In. exact Hv. }
  destruct Hread as (v & f & Hacc & Hv).
  pose proof (usages_accessor _ _ _ closedS v f y Hacc Hauy)
    as Huy.
  destruct (usages_back _ _ _ Hpay Huy) as [H | Huay];
    [ rewrite H in Hau; discriminate | ].
  assert (Hasy : s_any_source S y = false).
  { destruct (s_any_source S y) eqn:E; [ | reflexivity ].
    rewrite (cu0_any_source_read _ _ _ Hfix a y f v Huay E Hacc Hv)
      in Hcu0.
    discriminate. }
  assert (Hasx : s_any_source S x = false).
  { destruct (s_any_source S x) eqn:E; [ | reflexivity ].
    rewrite (any_source_fwd _ _ Hpxy E) in Hasy. discriminate. }
  assert (Hother : ∀ b, s_sources S x b = true -> s_has_source S b = true ->
                        b = a).
  { intros b Hb Hhb. destruct (node_eq_dec b a) as [-> | Hne];
      [ reflexivity | exfalso ].
    destruct (sources_fwd _ _ _ Hpxy Hb) as [H | Hyb];
      [ rewrite H in Hasy; discriminate | ].
    rewrite (cu0_other_source_read _ _ _ Hfix a y b f v Huay Hyb Hhb Hne
               Hacc Hv)
      in Hcu0.
    discriminate. }
  apply dom_iff. split; [ | split; [ exact Hhsa | ] ].
  - destruct (sources_fwd _ _ _ (path_one _ _ Hxr) Hsra)
      as [H | H]; [ rewrite H in Hasx; discriminate | exact H ].
  - intros [H | (b & c & Hb & Hhb & Hc & Hhc & Hne)];
      [ rewrite H in Hasx; discriminate | ].
    apply Hne. rewrite (Hother b Hb Hhb), (Hother c Hc Hhc). reflexivity.
Qed.

(* [r] given to [x]: an [r] with a source is dominated by what dominates
   [x]. *)
Lemma flow_unboxed (x r a : node) :
  s_alias S x r = true -> mem_pair a x dominated = true ->
  s_has_source S r = true -> mem_pair a r dominated = true.
Proof.
  intros Hxr Hx Hr. apply dom_iff in Hx. destruct Hx as (Hsxa & Hhsa & Hnm).
  assert (Hasx : s_any_source S x = false).
  { destruct (s_any_source S x) eqn:E; [ | reflexivity ].
    exfalso. apply Hnm. left. exact E. }
  assert (Hasr : s_any_source S r = false).
  { destruct (s_any_source S r) eqn:E; [ | reflexivity ].
    rewrite (any_source_alias _ _ _ closedS r x Hxr E)
      in Hasx.
    discriminate. }
  assert (Hto : ∀ s, s_sources S r s = true -> s_sources S x s = true).
  { intros s Hs.
    exact (sources_alias _ _ _ closedS r x s Hs Hasr Hxr
             Hasx). }
  assert (Hone : ∀ s, s_sources S x s = true -> s_has_source S s = true ->
                      s = a).
  { intros s Hs Hhs. destruct (node_eq_dec s a) as [-> | Hne];
      [ reflexivity | ].
    exfalso. apply Hnm. right. exists s, a. repeat split; assumption. }
  destruct (has_source_sources r Hr Hasr) as (s & Hs & Hhs).
  pose proof (Hone s (Hto s Hs) Hhs) as ->.
  apply dom_iff. split; [ exact Hs | split; [ exact Hhsa | ] ].
  intros [H | (b & c & Hb & Hhb & Hc & Hhc & Hne)];
    [ rewrite H in Hasr; discriminate | ].
  apply Hne. rewrite (Hone b (Hto b Hb) Hhb), (Hone c (Hto c Hc) Hhc).
  reflexivity.
Qed.

(* [r] given to [x], held by a witness of code that cannot change its
   calling convention: an [x] dominated by an allocation to unbox takes
   no such [r] with a source. *)
Lemma flow_cannot_change (x r a w codeid : node) (f : field) :
  s_alias S x r = true -> mem_pair a x dominated = true ->
  mem_node a to_unbox = true -> s_has_source S r = true ->
  (w, f, r) ∈ f_constructor F -> is_return_of_call f = true ->
  (w, Field_code_id_of_call_witness, codeid) ∈ f_constructor F ->
  mem_node codeid cannot_change = false.
Proof.
  intros Hxr Hx Ha Hr Hw Hf Hcid.
  pose proof (flow_unboxed x r a Hxr Hx Hr) as Hra.
  apply dom_iff in Hra. destruct Hra as (Hsra & Hhsa & _).
  destruct unboxed_rules as (crr0 & crr1 & crr & cu0 & cu & Huc & _ & Hto).
  destruct (mem_node codeid cannot_change) eqn:Ec; [ | reflexivity ].
  pose proof (Hto a Ha) as Hcu0.
  rewrite (cu0_result _ _ _ _ _ _ _ _ _ _ _ Huc r a f w codeid Hsra Hhsa Hw
             Hf Hcid Ec)
    in Hcu0.
  discriminate.
Qed.

End Least.

Theorem result_flow : result_flow_statement.
Proof.
  intros is_local F T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox x r _ Hleast Hdom Hunbox Hargs Hxr.
  split; [ | split ].
  - intros a Hr Ha Hx.
    exact (flow_kept is_local F T Hleast cannot_change max_unbox_size
             lambda_lifting dominated to_unbox Hdom Hunbox x r a Hargs Hxr Hr
             Ha Hx).
  - intros a Hx Hr.
    exact (flow_unboxed is_local F T Hleast dominated Hdom x r a Hxr Hx Hr).
  - intros a w codeid f Hx Ha Hr Hw Hf Hcid.
    exact (flow_cannot_change is_local F T Hleast cannot_change
             max_unbox_size lambda_lifting dominated to_unbox Hdom Hunbox x r a
             w codeid f Hxr Hx Ha Hr Hw Hf Hcid).
Qed.

(* ======================================================================== *)
(* The unboxed names and the decisions                                      *)
(* ======================================================================== *)

Lemma node_eqb_decides : decides_equality node_eqb.
Proof.
  intros n n'. split; [ apply node_eqb_eq | intros <-; apply node_eqb_refl ].
Qed.

Lemma lookup_node_some {A : Type} (n : node) (l : list (node * A)) :
  Keyed.lookup node_eqb n l <> None <-> n ∈ map fst l.
Proof.
  induction l as [ | [m a] l IH]; cbn [Keyed.lookup map fst].
  - split; [ intros H; exfalso; apply H; reflexivity | intros [] ].
  - destruct (node_eqb n m) eqn:E.
    + apply node_eqb_eq in E. subst m. split; [ intros _; left; reflexivity | ].
      intros _. discriminate.
    + rewrite IH. split; [ intros H; right; exact H | ].
      intros [-> | H]; [ rewrite node_eqb_refl in E; discriminate | exact H ].
Qed.

(* The names with unboxed fields are those that an allocation to unbox
   dominates. *)
Lemma unboxed_names (F : facts) (T : tables) (my : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (next : positive)
    (n : node) :
  n ∈ map fst (fst (unboxed_fields_of_names F T my dominated to_unbox next))
  <-> ∃ a, (a, n) ∈ dominated ∧ mem_node a to_unbox = true.
Proof.
  unfold unboxed_fields_of_names.
  assert (Hgen : ∀ l acc nx,
             n ∈ map fst
                   (fst (List.fold_left
                           (fun '(acc, next) '(a, x) =>
                              if mem_node a to_unbox
                                 && negb (existsb (fun '(y, _) => node_eqb x y)
                                            acc)
                              then
                                let '(fields, next) :=
                                  name_unboxed_fields F T my dominated
                                    to_unbox a x next
                                in
                                (acc ++ [(x, fields)], next)
                              else (acc, next))
                           l (acc, nx)))
             <-> n ∈ map fst acc
                 ∨ ∃ a, (a, n) ∈ l ∧ mem_node a to_unbox = true).
  { induction l as [ | [a x] l IH]; intros acc nx; cbn [List.fold_left].
    - split; [ intros H; left; exact H | ].
      intros [H | (a & [] & _)]. exact H.
    - destruct (mem_node a to_unbox
                && negb (existsb (fun '(y, _) => node_eqb x y) acc)) eqn:E.
      + destruct (name_unboxed_fields F T my dominated to_unbox a x nx)
          as [fields nx'].
        rewrite IH, map_app, in_app_iff. cbn [map fst In].
        apply andb_true_iff in E as [Ea _].
        split.
        * intros [[H | [<- | []]] | (a' & Ha' & Ht)].
          -- left. exact H.
          -- right. exists a. split; [ left; reflexivity | exact Ea ].
          -- right. exists a'. split; [ right; exact Ha' | exact Ht ].
        * intros [H | (a' & [Heq | Ha'] & Ht)].
          -- left. left. exact H.
          -- injection Heq as <- <-. left. right. left. reflexivity.
          -- right. exists a'. split; assumption.
      + rewrite IH. split.
        * intros [H | (a' & Ha' & Ht)]; [ left; exact H | right ].
          exists a'. split; [ right; exact Ha' | exact Ht ].
        * intros [H | (a' & [Heq | Ha'] & Ht)]; [ left; exact H | | ].
          -- injection Heq as Ha Hx. subst a' x. left. rewrite Ht in E.
             cbn [andb] in E. apply negb_false_iff in E.
             apply existsb_exists in E as ([y fy] & Hin & Hy).
             apply node_eqb_eq in Hy. subst y.
             apply in_map_iff. exists (n, fy). split; [ reflexivity | ].
             exact Hin.
          -- right. exists a'. split; assumption. }
  rewrite Hgen. split; [ intros [[] | H]; exact H | intros H; right; exact H ].
Qed.

Lemma unboxed_of_unit (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr)
    (n : node) :
  unboxed_of_node (unit_unboxed F T dominated to_unbox e) n <> None
  <-> ∃ a, (a, n) ∈ dominated ∧ mem_node a to_unbox = true.
Proof.
  unfold unboxed_of_node, unit_unboxed, unit_unboxed_fields.
  rewrite lookup_node_some. apply unboxed_names.
Qed.

Definition is_unbox_decision (d : decision) : bool :=
  match d with
  | Unbox _ => true
  | Decided _ | Keep_unused _ => false
  end.

Lemma nth_error_combine {A B : Type} (l : list A) (l' : list B) (i : nat)
    (a : A) (b : B) :
  nth_error (combine l l') i = Some (a, b)
  <-> nth_error l i = Some a ∧ nth_error l' i = Some b.
Proof.
  revert l' i. induction l as [ | x l IH]; intros [ | y l'] [ | i]; cbn;
    try (split; [ discriminate | intros [? ?]; discriminate ]).
  - split; [ intros [= -> ->]; auto | intros [[= ->] [= ->]]; reflexivity ].
  - exact (IH l' i).
Qed.

Lemma with_kinds_nth (params : list variable) (kinds : list flambda_kind)
    (i : nat) (x : variable) (kind : flambda_kind) :
  nth_error (with_kinds params kinds) i = Some (x, kind) ->
  nth_error params i = Some x ∧ (kind ∈ kinds ∨ kind = Value).
Proof.
  revert kinds i. induction params as [ | p params IH];
    intros [ | k kinds] [ | i]; cbn; try discriminate.
  - intros [= -> ->]. auto.
  - intros H. destruct (IH [] i H) as [H1 [[] | H2]]. auto.
  - intros [= -> ->]. split; [ reflexivity | left; left; reflexivity ].
  - intros H. destruct (IH kinds i H) as [H1 [H2 | H2]];
      split; auto; left; right; exact H2.
Qed.

Lemma with_kinds_length (params : list variable) (kinds : list flambda_kind) :
  length (with_kinds params kinds) = length params.
Proof.
  revert kinds. induction params as [ | p params IH]; intros [ | k kinds];
    cbn; auto.
Qed.

Section Decisions.

Variable sol : solution.
Variable unboxed : list (node * unboxed_fields).

(* What a decision says of the node of its parameter. *)
Definition decision_ok (d : decision) (n : node) : Prop :=
  (is_unbox_decision d = true <-> unboxed_of_node unboxed n <> None)
  ∧ ∀ kind, d = Decided (Keep kind) -> s_has_usage sol n = true.

Lemma is_var_used_plain (x : variable) (kind : flambda_kind) :
  plain_kind kind = true -> is_var_used sol x kind = true ->
  s_has_usage sol (Node_variable x) = true.
Proof. destruct kind; cbn; auto; discriminate. Qed.

Lemma is_node_used_plain (n : node) (kind : flambda_kind) :
  plain_kind kind = true -> is_node_used sol n kind = true ->
  s_has_usage sol n = true.
Proof. destruct kind; cbn; auto; discriminate. Qed.

Lemma handler_decisions_ok (is_exn : bool) (params : list variable)
    (metadata : handler_metadata) (body : expr) :
  forallb plain_kind (hm_params_kinds metadata) = true ->
  let decs := handler_decisions true sol unboxed is_exn
                (Handler params metadata body) in
  length decs = length params
  ∧ ∀ i d x, nth_error decs i = Some d -> nth_error params i = Some x ->
      decision_ok d (Node_variable x).
Proof.
  intros Hplain decs. unfold decs, handler_decisions.
  set (bound := with_kinds params (hm_params_kinds metadata)).
  split.
  - rewrite length_map, length_combine. unfold params_to_keep.
    rewrite length_map, Nat.min_id. apply with_kinds_length.
  - intros i d x Hd Hx. rewrite nth_error_map in Hd.
    destruct (nth_error (combine bound _) i) as [[[y kind] keep] | ] eqn:E;
      [ | discriminate ].
    cbn in Hd. injection Hd as <-.
    apply nth_error_combine in E as [Eb Ek].
    destruct (with_kinds_nth _ _ _ _ _ Eb) as [Ey Hkind].
    rewrite Ey in Hx. injection Hx as ->.
    assert (Hpk : plain_kind kind = true).
    { destruct Hkind as [Hin | ->]; [ | reflexivity ].
      rewrite forallb_forall in Hplain. exact (Hplain _ Hin). }
    split.
    + destruct (unboxed_of_node unboxed (Node_variable x)); cbn.
      * split; [ intros _; discriminate | reflexivity ].
      * split; [ | intros H; exfalso; apply H; reflexivity ].
        destruct (negb (is_var_used sol x kind)), keep; discriminate.
    + intros kind' Hk.
      destruct (unboxed_of_node unboxed (Node_variable x)); [ discriminate | ].
      destruct keep; cbn in Hk; [ | discriminate ].
      destruct (is_var_used sol x kind) eqn:Eu; [ | discriminate ].
      exact (is_var_used_plain _ _ Hpk Eu).
Qed.

Lemma node_decisions_ok (fixed_arity : bool) (nodes : list node)
    (kinds : list flambda_kind) :
  forallb plain_kind kinds = true ->
  let decs := if fixed_arity then fixed_arity_decisions sol unboxed nodes kinds
              else decisions_u sol unboxed nodes kinds in
  ∀ i d n, nth_error decs i = Some d -> nth_error nodes i = Some n ->
    decision_ok d n.
Proof.
  intros Hplain decs i d n Hd Hn.
  assert (Hgen : ∀ n' kind,
            nth_error (combine nodes kinds) i = Some (n', kind) ->
            d = (if fixed_arity
                 then match unboxed_of_node unboxed n' with
                      | Some fields => Unbox fields
                      | None =>
                        if is_node_used sol n' kind then Decided (Keep kind)
                        else Keep_unused kind
                      end
                 else match unboxed_of_node unboxed n' with
                      | Some fields => Unbox fields
                      | None =>
                        Decided (if is_node_used sol n' kind then Keep kind
                                 else Delete)
                      end) ->
            decision_ok d n).
  { intros n' kind E Heq. apply nth_error_combine in E as [En Ek].
    rewrite En in Hn. injection Hn as <-.
    assert (Hpk : plain_kind kind = true).
    { rewrite forallb_forall in Hplain. apply Hplain.
      exact (nth_error_In _ _ Ek). }
    subst d. split.
    - destruct fixed_arity, (unboxed_of_node unboxed n'); cbn;
        try (split; [ intros _; discriminate | reflexivity ]);
        (split; [ | intros H; exfalso; apply H; reflexivity ]);
        destruct (is_node_used sol n' kind); discriminate.
    - intros kind' Hk. destruct fixed_arity, (unboxed_of_node unboxed n');
        try discriminate;
        destruct (is_node_used sol n' kind) eqn:Eu; try discriminate;
        exact (is_node_used_plain _ _ Hpk Eu). }
  unfold decs, fixed_arity_decisions, decisions_u in Hd.
  destruct fixed_arity; rewrite nth_error_map in Hd;
    destruct (nth_error (combine nodes kinds) i) as [[n' kind] | ] eqn:E;
    try discriminate; cbn in Hd; injection Hd as Hd;
    exact (Hgen n' kind eq_refl (eq_sym Hd)).
Qed.

End Decisions.

(* ======================================================================== *)
(* The facts of the traversal                                               *)
(* ======================================================================== *)

(* What FlambdaReaperFlowProofs.traverse_unit_facts says, with the code
   deps of [unit_code_deps]. *)
Lemma unit_facts (flag : reaper_preserve_direct_calls) (u : unit_interface)
    (e : expr) (slots_read : Prop) :
  in_flow_fragment e = true ->
  slots_read ∨ projection_free e = true ->
  ∃ (noted : symbol -> Prop) (deps : closure_dep -> Prop),
    FlambdaReaperFlowProofs.expr_edges
      (graph_of_facts (traverse_unit flag u e)) noted deps
      (unit_code_deps flag u e) (unit_kept e) (unit_codes e)
      (static_names symbols_of_binding e) slots_read False None
      (unit_conts u) e
    ∧ g_any_source (graph_of_facts (traverse_unit flag u e)) Node_all_constants
    ∧ (∀ (k : continuation) (p : node),
         p ∈ unit_cont_nodes u k ->
         g_any_usage (graph_of_facts (traverse_unit flag u e)) p)
    ∧ NoDup (map fst (unit_code_deps flag u e))
    ∧ (∀ id : code_id,
         id ∈ map fst (unit_code_deps flag u e) ->
         id ∈ static_names code_ids_of_binding e).
Proof.
  intros Hfrag Hslots. unfold in_flow_fragment in Hfrag.
  apply andb_true_iff in Hfrag as [Hfrag Hdist].
  rewrite FlambdaReaperFlowProofs.traverse_unit_eq.
  destruct (proj1 (FlambdaReaperFlowProofs.traverse_makes (unit_codes e)
                     (static_names symbols_of_binding e)
                     (projected_value_slots e) slots_read false) e Hfrag Hdist
              (fun w Hw => proj2 (FlambdaMapProofs.among_In w _) Hw) Hslots
              (fun x Hx => Hx)
              (FlambdaReaperFlowProofs.initial_env flag u e) eq_refl
              FlambdaReaperFlowProofs.initial_acc)
    as [[Hle4 (l & Hl & Hperm)] Hedges].
  set (a4 := traverse (FlambdaReaperFlowProofs.initial_env flag u e) e
               FlambdaReaperFlowProofs.initial_acc) in *.
  set (a5 := add_fact (Fact_any_usage (Node_made 1)) a4).
  destruct (FlambdaReaperFlowProofs.resolve_delayed_deps_spec
              (static_names symbols_of_binding e) a5)
    as (S6 & _ & _ & _ & _).
  set (a6 := resolve_delayed_deps _ a5) in *.
  pose proof (FlambdaReaperFlowProofs.steady_trans _ _ _
                (FlambdaReaperFlowProofs.add_facts_steady _ a4) S6) as S46.
  assert (Hcds : acc_code_deps a6 = l).
  { rewrite (proj2 S46), Hl, app_nil_r. reflexivity. }
  change (unit_code_deps flag u e) with (acc_code_deps a6).
  exists (FlambdaReaperFlowProofs.acc_noted a6),
    (FlambdaReaperFlowProofs.acc_deps a6).
  split; [ | split; [ | split; [ | split ] ] ].
  - exact (Hedges a6 (proj1 S46)).
  - apply FlambdaReaperFlowProofs.facts_any_source.
    apply (proj1 S46), Hle4. right. left. reflexivity.
  - intros k p Hp. unfold unit_cont_nodes, unit_conts in Hp.
    simpl in Hp. apply FlambdaReaperFlowProofs.facts_any_usage.
    destruct (Pos.eqb k (unit_exn u));
      [ | destruct (Pos.eqb k (unit_return u)) ];
      simpl in Hp; [ | | contradiction ]; destruct Hp as [<- | []].
    + apply (proj1 S46), Hle4. left. reflexivity.
    + apply (proj1 S6). left. reflexivity.
  - rewrite Hcds.
    apply (Permutation.Permutation_NoDup (Permutation.Permutation_sym Hperm)).
    apply FlambdaMapProofs.distinct_NoDup. exact Hdist.
  - intros id Hid. rewrite Hcds in Hid.
    exact (Permutation.Permutation_in _ Hperm Hid).
Qed.

(* The code of a unit: where its code ids are bound, and its metadata. *)
Lemma unit_codes_names :
  (∀ e : expr, ∀ id : code_id,
     id ∈ static_names code_ids_of_binding e ->
     (∃ c, (id, c) ∈ unit_codes e) ∧ id ∈ static_names code_ids_bound e)
  ∧ (∀ h : continuation_handler, ∀ id : code_id,
       id ∈ handler_static_names code_ids_of_binding h ->
       (∃ c, (id, c) ∈ handler_unit_codes h)
       ∧ id ∈ handler_static_names code_ids_bound h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind; cbn [static_names
    handler_static_names unit_codes handler_unit_codes].
  - intros x kind d body IH id H. exact (IH id H).
  - intros bindings body _ IH id H. apply in_app_or in H as [H | H].
    + apply in_flat_map in H as (b & Hb & Hid).
      destruct b as [id' c | id' | s const]; cbn in Hid;
        [ | destruct Hid | destruct Hid ].
      destruct Hid as [-> | []]. split.
      * exists c. apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id c). split; [ exact Hb | left; reflexivity ].
      * apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id c). split; [ exact Hb | left; reflexivity ].
    + destruct (IH id H) as [(c & Hc) Hb]. split.
      * exists c. apply in_or_app. right. exact Hc.
      * apply in_or_app. right. exact Hb.
  - intros k h body IHh IH id H. apply in_app_or in H as [H | H].
    + destruct (IHh id H) as [(c & Hc) Hb].
      split; [ exists c | ]; apply in_or_app; left; assumption.
    + destruct (IH id H) as [(c & Hc) Hb].
      split; [ exists c | ]; apply in_or_app; right; assumption.
  - intros handlers body IHh IH id H. apply in_app_or in H as [H | H].
    + apply in_flat_map in H as ([k h] & Hkh & Hid).
      destruct (IHh k h Hkh id Hid) as [(c & Hc) Hb].
      split; [ exists c | ]; apply in_or_app; left; apply in_flat_map;
        exists (k, h); split; assumption.
    + destruct (IH id H) as [(c & Hc) Hb].
      split; [ exists c | ]; apply in_or_app; right; assumption.
  - intros ap id [].
  - intros ac id [].
  - intros scrutinee arms id [].
  - intros id [].
  - intros params metadata handler IH id H. exact (IH id H).
Qed.

Lemma bound_code_metadata_codes :
  (∀ e : expr,
     bound_code_metadata e
     = map (fun '(id, c) => (id, metadata_of_code c)) (unit_codes e))
  ∧ (∀ h : continuation_handler,
       handler_bound_code_metadata h
       = map (fun '(id, c) => (id, metadata_of_code c))
           (handler_unit_codes h)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind; cbn [bound_code_metadata
    handler_bound_code_metadata unit_codes handler_unit_codes].
  - intros x kind d body IH. exact IH.
  - intros bindings body _ IH. rewrite map_app, IH. f_equal.
    induction bindings as [ | b bindings IHb]; [ reflexivity | ].
    cbn [flat_map]. rewrite map_app, IHb.
    destruct b; reflexivity.
  - intros k h body IHh IH. rewrite map_app, IHh, IH. reflexivity.
  - intros handlers body IHh IH. rewrite map_app, IH. f_equal.
    induction handlers as [ | [k h] handlers IHhs]; [ reflexivity | ].
    cbn [flat_map]. rewrite map_app, (IHh k h (or_introl eq_refl)).
    f_equal. apply IHhs. intros k' h' Hin. apply (IHh k' h'). right.
    exact Hin.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH. exact IH.
Qed.

(* The edges of the code of a unit. *)
Lemma unit_codes_edges (G : graph) (noted : symbol -> Prop)
    (deps : closure_dep -> Prop) (cds : finite_map code_dep)
    (keep : value_slot -> bool) (codes : list (code_id * code))
    (symbols : list symbol) (slots_read heap_prims : Prop) :
  (∀ e : expr, ∀ ctx ce,
     FlambdaReaperFlowProofs.expr_edges G noted deps cds keep codes symbols
       slots_read heap_prims ctx ce e ->
     ∀ id c, (id, c) ∈ unit_codes e ->
       FlambdaReaperFlowProofs.code_edges G noted deps cds keep codes symbols
         slots_read heap_prims id c)
  ∧ (∀ h : continuation_handler, ∀ ctx ce,
       FlambdaReaperFlowProofs.handler_edges G noted deps cds keep codes
         symbols slots_read heap_prims ctx ce h ->
       ∀ id c, (id, c) ∈ handler_unit_codes h ->
         FlambdaReaperFlowProofs.code_edges G noted deps cds keep codes
           symbols slots_read heap_prims id c).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [FlambdaReaperFlowProofs.expr_edges
         FlambdaReaperFlowProofs.handler_edges unit_codes handler_unit_codes].
  - intros x kind d body IH ctx ce [_ H] id c Hin. exact (IH _ _ H id c Hin).
  - intros bindings body _ IH ctx ce [Hb H] id c Hin.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH _ _ H id c Hin) ].
    apply in_flat_map in Hin as (b & Hbin & Hin).
    destruct b as [id' c' | id' | s const]; cbn in Hin;
      [ | destruct Hin | destruct Hin ].
    destruct Hin as [[= <- <-] | []].
    exact (proj2 (proj1 (FlambdaMapProofs.every_in _ _ _) Hb _ Hbin)).
  - intros k h body IHh IH ctx ce [Hh H] id c Hin.
    apply in_app_or in Hin as [Hin | Hin];
      [ exact (IHh _ _ Hh id c Hin) | exact (IH _ _ H id c Hin) ].
  - intros handlers body IHh IH ctx ce [Hhs H] id c Hin.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH _ _ H id c Hin) ].
    apply in_flat_map in Hin as ([k h] & Hkh & Hin).
    exact (IHh k h Hkh _ _
             (proj1 (FlambdaMapProofs.every_in _ _ _) Hhs _ Hkh) id c Hin).
  - intros ap ctx ce _ id c [].
  - intros ac ctx ce _ id c [].
  - intros scrutinee arms ctx ce _ id c [].
  - intros ctx ce _ id c [].
  - intros params metadata handler IH ctx ce H id c Hin.
    exact (IH _ _ H id c Hin).
Qed.

(* ======================================================================== *)
(* The code that a call is known to call                                    *)
(* ======================================================================== *)

Lemma forallb_false_exists {A : Type} (f : A -> bool) (l : list A) :
  forallb f l = false -> ∃ x, x ∈ l ∧ f x = false.
Proof.
  induction l as [ | y l IH]; cbn; [ discriminate | ].
  destruct (f y) eqn:E; cbn.
  - intros H. destruct (IH H) as (x & Hx & Hf). exists x. auto.
  - intros _. exists y. auto.
Qed.

Lemma pos_eqb_decides : decides_equality Pos.eqb.
Proof. intros p q. apply Pos.eqb_eq. Qed.

Lemma singleton_some (ids : list code_id) (id x : code_id) :
  Keyed.get_singleton Pos.eqb ids = Some id -> x ∈ ids -> x = id.
Proof.
  intros H0 Hx.
  pose proof (proj1 (FlambdaMapProofs.keyed_get_singleton _ Pos.eqb
                       pos_eqb_decides id ids) H0 x) as H.
  rewrite (proj2 (FlambdaMapProofs.keyed_among_in _ Pos.eqb pos_eqb_decides
                    x ids) Hx)
    in H.
  symmetry in H. apply Pos.eqb_eq in H. exact H.
Qed.

Lemma singleton_none (ids : list code_id) (x : code_id) :
  Keyed.get_singleton Pos.eqb ids = None -> x ∈ ids ->
  ∃ y, y ∈ ids ∧ y <> x.
Proof.
  intros H Hx. destruct (forallb (fun y => Pos.eqb y x) ids) eqn:E.
  - exfalso.
    assert (H' : Keyed.get_singleton Pos.eqb ids = Some x);
      [ | rewrite H in H'; discriminate ].
    apply (proj2 (FlambdaMapProofs.keyed_get_singleton _ Pos.eqb
                    pos_eqb_decides x ids)).
    intros key'. destruct (Keyed.among Pos.eqb key' ids) eqn:Ea.
    + apply (FlambdaMapProofs.keyed_among_in _ Pos.eqb pos_eqb_decides)
        in Ea.
      rewrite forallb_forall in E. symmetry. exact (E _ Ea).
    + destruct (Pos.eqb key' x) eqn:Eb; [ | reflexivity ].
      apply Pos.eqb_eq in Eb. subst key'.
      rewrite (proj2 (FlambdaMapProofs.keyed_among_in _ Pos.eqb
                        pos_eqb_decides x ids) Hx)
        in Ea.
      discriminate.
  - destruct (forallb_false_exists _ _ E) as (y & Hy & Hne).
    exists y. split; [ exact Hy | ]. intros ->. rewrite Pos.eqb_refl in Hne.
    discriminate.
Qed.

Section Callee.

Variable F : facts.
Variable sol : solution.

Lemma called_some (ap : apply_expr) (id : code_id) :
  code_id_actually_called F sol ap = Some id ->
  ∃ ids,
    code_id_actually_directly_called F sol (node_of_simple (ap_callee ap))
    = Some ids
    ∧ Keyed.get_singleton Pos.eqb ids = Some id.
Proof.
  unfold code_id_actually_called.
  destruct (ap_call_kind ap); try discriminate;
    destruct (ap_callee ap); try discriminate;
    destruct (code_id_actually_directly_called F sol _) as [ids | ];
    try discriminate; intros H; exists ids; split; auto.
Qed.

Lemma called_none (ap : apply_expr) :
  ap_call_kind ap <> Indirect_unknown_arity ->
  code_id_actually_called F sol ap = None ->
  (∃ c, ap_callee ap = Const c)
  ∨ code_id_actually_directly_called F sol (node_of_simple (ap_callee ap))
    = None
  ∨ ∃ ids,
      code_id_actually_directly_called F sol (node_of_simple (ap_callee ap))
      = Some ids
      ∧ Keyed.get_singleton Pos.eqb ids = None.
Proof.
  intros Hk. unfold code_id_actually_called.
  destruct (ap_call_kind ap); [ | exfalso; apply Hk; reflexivity | ];
    (destruct (ap_callee ap) as [x | sym | c];
       [ | | left; exists c; reflexivity ];
     destruct (code_id_actually_directly_called F sol _) as [ids | ];
     intros H; right; [ right; exists ids; auto | left; reflexivity
                      | right; exists ids; auto | left; reflexivity ]).
Qed.

Lemma directly_member (f : node) (ids : list code_id) (a s : node)
    (x : code_id) :
  code_id_actually_directly_called F sol f = Some ids ->
  (a, known, f) ∈ f_accessor F -> s_sources sol a s = true ->
  s_has_source sol s = true ->
  (s, Field_code_id_of_call_witness, Node_code_id x) ∈ f_constructor F ->
  x ∈ ids.
Proof.
  intros H Ha Hs Hhs Hc. unfold code_id_actually_directly_called in H.
  destruct (existsb _ _); [ discriminate | ]. injection H as <-.
  apply in_flat_map. exists (a, known, f). split.
  - apply filter_In. split; [ exact Ha | ].
    rewrite node_eqb_refl, field_eqb_refl. reflexivity.
  - apply in_flat_map.
    exists (s, Field_code_id_of_call_witness, Node_code_id x).
    split; [ exact Hc | ]. rewrite field_eqb_refl, Hs, Hhs. left. reflexivity.
Qed.

Lemma directly_inv (f : node) (ids : list code_id) (x : code_id) :
  code_id_actually_directly_called F sol f = Some ids -> x ∈ ids ->
  ∃ a s, (a, known, f) ∈ f_accessor F ∧ s_sources sol a s = true
         ∧ s_has_source sol s = true
         ∧ (s, Field_code_id_of_call_witness, Node_code_id x)
             ∈ f_constructor F.
Proof.
  intros H Hx. unfold code_id_actually_directly_called in H.
  destruct (existsb _ _); [ discriminate | ]. injection H as <-.
  apply in_flat_map in Hx as ([[a fa] base] & Ha & Hx).
  apply filter_In in Ha as [Ha Ht]. apply andb_true_iff in Ht as [Hb Hf].
  apply node_eqb_eq in Hb. apply field_eqb_eq in Hf. subst base fa.
  apply in_flat_map in Hx as ([[s fs] c] & Hc & Hx).
  destruct c as [ | | id | | ]; cbn in Hx; try destruct Hx.
  destruct (field_eqb fs Field_code_id_of_call_witness && s_sources sol a s
            && s_has_source sol s) eqn:E; [ | destruct Hx ].
  destruct Hx as [<- | []].
  apply andb_true_iff in E as [E Hhs]. apply andb_true_iff in E as [Hf Hs].
  apply field_eqb_eq in Hf. subst fs.
  exists a, s. auto.
Qed.

Lemma directly_no_any (f : node) (ids : list code_id) (a : node) :
  code_id_actually_directly_called F sol f = Some ids ->
  (a, known, f) ∈ f_accessor F -> s_any_source sol a = false.
Proof.
  intros H Ha. unfold code_id_actually_directly_called in H.
  destruct (existsb _ _) eqn:E; [ discriminate | ].
  destruct (s_any_source sol a) eqn:Ea; [ | reflexivity ].
  rewrite <- E. symmetry. apply existsb_exists. exists (a, known, f).
  split; [ | exact Ea ]. apply filter_In. split; [ exact Ha | ].
  rewrite node_eqb_refl, field_eqb_refl. reflexivity.
Qed.

Lemma directly_none (f : node) :
  code_id_actually_directly_called F sol f = None ->
  ∃ a, (a, known, f) ∈ f_accessor F ∧ s_any_source sol a = true.
Proof.
  intros H. unfold code_id_actually_directly_called in H.
  destruct (existsb _ _) eqn:E; [ | discriminate ].
  apply existsb_exists in E as ([[a fa] base] & Ha & Hany).
  apply filter_In in Ha as [Ha Ht]. apply andb_true_iff in Ht as [Hb Hf].
  apply node_eqb_eq in Hb. apply field_eqb_eq in Hf. subst base fa.
  exists a. auto.
Qed.

End Callee.

(* ======================================================================== *)
(* The checks                                                               *)
(* ======================================================================== *)

Section Checks.

Variable F : facts.
Variable T : tables.

Abbreviation S := (solution_of_tables T).

(* The nodes read off one callee with the field of known arity have the
   same sources. *)
Lemma widgets_same (a a' f : node) :
  FlambdaReaperCallingConventionsSpec.check_apply_widgets F T = true ->
  (a, known, f) ∈ f_accessor F -> (a', known, f) ∈ f_accessor F ->
  s_any_source S a = s_any_source S a'
  ∧ ∀ y, s_sources S a y = true -> s_sources S a' y = true.
Proof.
  intros H Ha Ha'.
  unfold FlambdaReaperCallingConventionsSpec.check_apply_widgets in H.
  rewrite forallb_forall in H. specialize (H _ Ha). cbn beta iota in H.
  rewrite forallb_forall in H. specialize (H _ Ha'). cbn beta iota in H.
  rewrite field_eqb_refl, node_eqb_refl in H. cbn [negb andb orb] in H.
  apply andb_true_iff in H as [Hany Hs]. split.
  - apply Bool.eqb_prop in Hany. exact Hany.
  - intros y Hy. rewrite forallb_forall in Hs.
    specialize (Hs (a, y) (proj1 (mem_pair_In _ _ _) Hy)). cbn beta iota in Hs.
    rewrite node_eqb_refl in Hs. exact Hs.
Qed.

Section Known.

Variable cds : finite_map code_dep.
Hypothesis Hknown : check_known_witnesses F T cds = true.

Lemma kw_source (a f s : node) :
  (a, known, f) ∈ f_accessor F -> s_sources S a s = true ->
  s_has_source S s = true ->
  ∃ id cd, (id, cd) ∈ cds ∧ s = cd_known_arity_call_witness cd.
Proof.
  intros Ha Hs Hhs. unfold check_known_witnesses in Hknown.
  apply andb_true_iff in Hknown as [H _].
  rewrite forallb_forall in H. specialize (H _ Ha). cbn beta iota in H.
  rewrite field_eqb_refl in H. cbn [negb orb] in H.
  rewrite forallb_forall in H.
  specialize (H (a, s) (proj1 (mem_pair_In _ _ _) Hs)). cbn beta iota in H.
  rewrite node_eqb_refl in H. cbn in Hhs. rewrite Hhs in H.
  cbn [negb orb] in H. apply existsb_exists in H as ([id cd] & Hin & Heq).
  apply node_eqb_eq in Heq. exists id, cd. auto.
Qed.

Lemma kw_constructor (id : code_id) (cd : code_dep) (f : field) (r : node) :
  (id, cd) ∈ cds -> (cd_known_arity_call_witness cd, f, r) ∈ f_constructor F ->
  (∀ i, f = Field_return_of_call (Normal i) ->
        nth_error (cd_return cd) i = Some r)
  ∧ (f = Field_code_id_of_call_witness -> r = Node_code_id id).
Proof.
  intros Hin Hc. unfold check_known_witnesses in Hknown.
  apply andb_true_iff in Hknown as [_ H].
  rewrite forallb_forall in H. specialize (H _ Hin). cbn beta iota in H.
  rewrite forallb_forall in H. specialize (H _ Hc). cbn beta iota in H.
  rewrite node_eqb_refl in H. cbn [negb orb] in H. split.
  - intros i ->. destruct (nth_error (cd_return cd) i) as [r' | ];
      [ | discriminate ].
    apply node_eqb_eq in H. subst r'. reflexivity.
  - intros ->. apply node_eqb_eq in H. exact H.
Qed.

End Known.

(* A witness of code that cannot change its calling convention for the
   rules: its code is in a closed table. *)
Lemma cc_closed_witness (cannot_change : list node) (w codeid : node) :
  check_cc_closed F T cannot_change = true ->
  (w, Field_code_id_of_call_witness, codeid) ∈ f_constructor F ->
  s_has_usage S w = true ->
  cannot_change_witness_calling_convention F T w = true ->
  mem_node codeid cannot_change = true.
Proof.
  intros Hcc Hc Hu Hw. unfold check_cc_closed in Hcc.
  apply andb_true_iff in Hcc as [H _]. unfold check_code_id_of_witness in H.
  rewrite forallb_forall in H. apply H.
  unfold derived_by_code_id_of_witness. apply in_flat_map.
  exists (w, Field_code_id_of_call_witness, codeid). split; [ exact Hc | ].
  rewrite field_eqb_refl. cbn in Hu. rewrite Hu, Hw. left. reflexivity.
Qed.

Lemma unknown_widgets_same (a a' f : node) :
  check_unknown_widgets F T = true ->
  (a, unknown, f) ∈ f_accessor F -> (a', unknown, f) ∈ f_accessor F ->
  s_any_source S a = s_any_source S a'
  ∧ ∀ y, s_sources S a y = true -> s_sources S a' y = true.
Proof.
  intros H Ha Ha'. unfold check_unknown_widgets in H.
  rewrite forallb_forall in H. specialize (H _ Ha). cbn beta iota in H.
  rewrite forallb_forall in H. specialize (H _ Ha'). cbn beta iota in H.
  rewrite field_eqb_refl, node_eqb_refl in H. cbn [negb andb orb] in H.
  apply andb_true_iff in H as [Hany Hs]. split.
  - apply Bool.eqb_prop in Hany. exact Hany.
  - intros y Hy. rewrite forallb_forall in Hs.
    specialize (Hs (a, y) (proj1 (mem_pair_In _ _ _) Hy)). cbn beta iota in Hs.
    rewrite node_eqb_refl in Hs. exact Hs.
Qed.

Lemma unknown_source (a f s : node) :
  check_unknown_witnesses F T = true ->
  (a, unknown, f) ∈ f_accessor F -> s_sources S a s = true ->
  s_has_source S s = true ->
  (∃ b, (b, unknown, s) ∈ f_constructor F)
  ∧ ∃ codeid, (s, Field_code_id_of_call_witness, codeid) ∈ f_constructor F.
Proof.
  intros H Ha Hs Hhs. unfold check_unknown_witnesses in H.
  rewrite forallb_forall in H. specialize (H _ Ha). cbn beta iota in H.
  rewrite field_eqb_refl in H. cbn [negb orb] in H.
  rewrite forallb_forall in H.
  specialize (H (a, s) (proj1 (mem_pair_In _ _ _) Hs)). cbn beta iota in H.
  rewrite node_eqb_refl in H. cbn in Hhs. rewrite Hhs in H.
  cbn [negb orb] in H. apply andb_true_iff in H as [H1 H2]. split.
  - apply existsb_exists in H1 as ([[b fb] s'] & Hin & Ht).
    apply andb_true_iff in Ht as [Hf Hs']. apply field_eqb_eq in Hf.
    apply node_eqb_eq in Hs'. subst fb s'. exists b. exact Hin.
  - apply existsb_exists in H2 as ([[s' fc] c] & Hin & Ht).
    apply andb_true_iff in Ht as [Hs' Hf]. apply field_eqb_eq in Hf.
    apply node_eqb_eq in Hs'. subst fc s'. exists c. exact Hin.
Qed.

(* A witness held in the field of unknown arity with a use: its code is
   in a closed table. *)
Lemma cc_closed_unknown (cannot_change : list node) (b w codeid : node) :
  check_cc_closed F T cannot_change = true ->
  (b, unknown, w) ∈ f_constructor F -> s_has_usage S w = true ->
  (w, Field_code_id_of_call_witness, codeid) ∈ f_constructor F ->
  mem_node codeid cannot_change = true.
Proof.
  intros Hcc Hb Hu Hc. unfold check_cc_closed in Hcc.
  apply andb_true_iff in Hcc as [_ H]. unfold check_code_id_unknown_arity in H.
  rewrite forallb_forall in H. apply H.
  unfold derived_by_code_id_unknown_arity. apply in_flat_map.
  exists (b, unknown, w). split; [ exact Hb | ].
  rewrite field_eqb_refl. cbn in Hu. rewrite Hu. cbn [andb].
  unfold code_ids_of_witness. apply in_flat_map.
  exists (w, Field_code_id_of_call_witness, codeid). split; [ exact Hc | ].
  rewrite node_eqb_refl, field_eqb_refl. left. reflexivity.
Qed.

End Checks.

(* ======================================================================== *)
(* The calls of a unit                                                      *)
(* ======================================================================== *)

Lemma decisions_u_unbox (sol : solution)
    (unboxed : list (node * unboxed_fields))
    (nodes : list node) (kinds : list flambda_kind) (i : nat) (d : decision) :
  nth_error (decisions_u sol unboxed nodes kinds) i = Some d ->
  ∃ n, nth_error nodes i = Some n
       ∧ (is_unbox_decision d = true <-> unboxed_of_node unboxed n <> None).
Proof.
  intros Hd. unfold decisions_u in Hd. rewrite nth_error_map in Hd.
  destruct (nth_error (combine nodes kinds) i) as [[n kind] | ] eqn:E;
    [ | discriminate ].
  apply nth_error_combine in E as [En _]. exists n. split; [ exact En | ].
  cbn in Hd. injection Hd as <-.
  destruct (unboxed_of_node unboxed n); cbn.
  - split; [ intros _; discriminate | reflexivity ].
  - split; [ | intros H; exfalso; apply H; reflexivity ].
    destruct (is_node_used sol n kind); discriminate.
Qed.

Lemma lookup_counts (ce : finite_map (list node)) (k : continuation) :
  lookup k (FlambdaReaperFlowProofs.counts ce)
  = option_map (@length node) (lookup k ce).
Proof.
  induction ce as [ | [k' ns] ce IH]; [ reflexivity | ].
  unfold FlambdaReaperFlowProofs.counts in *. cbn [map lookup Keyed.lookup].
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma return_not_local (is_local : field -> bool) (r : return_kind) :
  is_local_slots_only is_local -> is_local (Field_return_of_call r) = false.
Proof.
  intros Hloc. destruct (is_local (Field_return_of_call r)) eqn:E;
    [ | reflexivity ].
  destruct (Hloc _ E) as [(w & Hw) | (fs & Hfs)]; discriminate.
Qed.

Lemma known_not_local (is_local : field -> bool) :
  is_local_slots_only is_local -> is_local known = false.
Proof.
  intros Hloc. destruct (is_local known) eqn:E; [ | reflexivity ].
  destruct (Hloc _ E) as [(w & Hw) | (fs & Hfs)]; discriminate.
Qed.

Lemma code_id_not_local (is_local : field -> bool) :
  is_local_slots_only is_local ->
  is_local Field_code_id_of_call_witness = false.
Proof.
  intros Hloc. destruct (is_local Field_code_id_of_call_witness) eqn:E;
    [ | reflexivity ].
  destruct (Hloc _ E) as [(w & Hw) | (fs & Hfs)]; discriminate.
Qed.

Lemma group_by_sizes_length {A : Type} (sizes : list nat) (l : list A) :
  length (group_by_sizes sizes l) = length sizes.
Proof.
  revert l. induction sizes as [ | n sizes IH]; intros l; cbn; auto.
Qed.

(* A call whose arity has at most one parameter has one group of
   arguments. *)
Lemma one_group (ap : apply_expr) :
  length (ap_args_arity ap) <= 1 ->
  length (FlambdaReaperFlowProofs.call_groups ap) = 1.
Proof.
  intros H. unfold FlambdaReaperFlowProofs.call_groups, group_by_parameter.
  pose proof (group_by_sizes_length
                (map (@length flambda_kind)
                   (unarize_per_parameter (ap_args_arity ap)))
                (ap_args ap)) as E.
  rewrite length_map in E. unfold unarize_per_parameter in E.
  rewrite length_map in E.
  destruct (group_by_sizes _ _) as [ | g [ | g' gs]]; cbn in E |- *;
    [ reflexivity | reflexivity | lia ].
Qed.

Section Unit.

Variable flag : reaper_preserve_direct_calls.
Variable is_local : field -> bool.
Variable u : unit_interface.
Variable e0 : expr.
Variable T : tables.
Variable cannot_change : list node.
Variable max_unbox_size : nat.
Variable lambda_lifting : bool.
Variable dominated : list (node * node).
Variable to_unbox : list node.

Abbreviation F := (traverse_unit flag u e0).
Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).
Abbreviation cds := (unit_code_deps flag u e0).
Abbreviation unboxed := (unit_unboxed F T dominated to_unbox e0).

Hypothesis Hloc : is_local_slots_only is_local.
Hypothesis Hleast : least_solution is_local F T.
Hypothesis Hcc : check_cc_closed F T cannot_change = true.
Hypothesis Hdom : check_dominated T dominated = true.
Hypothesis Hunbox :
  check_to_unbox_fixed is_local F T cannot_change max_unbox_size
    lambda_lifting to_unbox
  = true.
Hypothesis Hwidgets :
  FlambdaReaperCallingConventionsSpec.check_apply_widgets F T = true.
Hypothesis Hcodes : check_used_code_ids T = true.
Hypothesis Hargs : check_argument_reads F = true.
Hypothesis Hknown : check_known_witnesses F T cds = true.

Variable slots_read : Prop.
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Hypothesis Hconsts : g_any_source G Node_all_constants.
Hypothesis Hnodup : NoDup (map fst cds).
Hypothesis Hids :
  ∀ id, id ∈ map fst cds -> id ∈ static_names code_ids_of_binding e0.

Abbreviation code_edges :=
  (FlambdaReaperFlowProofs.code_edges G noted deps cds (unit_kept e0)
     (unit_codes e0) (static_names symbols_of_binding e0) slots_read False).

Hypothesis Hcode_edges : ∀ id c, (id, c) ∈ unit_codes e0 -> code_edges id c.

Lemma closedU : closed is_local G S.
Proof. exact (closedS is_local F T Hleast). Qed.

Lemma has_usage_back (x y : node) :
  path T x y -> s_has_usage S y = true -> s_has_usage S x = true.
Proof.
  intros H. induction H as [ | x y' z Ha _ IH]; intros Hu; [ exact Hu | ].
  exact (has_usage_alias _ _ _ closedU y' x (IH Hu) Ha).
Qed.

Lemma to_unbox_any_usage (a : node) :
  mem_node a to_unbox = true -> s_any_usage S a = false.
Proof.
  intros Ha.
  destruct (unboxed_rules is_local F T cannot_change max_unbox_size
              lambda_lifting to_unbox Hunbox)
    as (crr0 & crr1 & crr & cu0 & cu & Huc & _ & Hto).
  pose proof (Hto a Ha) as Hcu0.
  destruct (s_any_usage S a) eqn:E; [ | reflexivity ].
  rewrite (cu0_any_usage _ _ _ _ _ _ _ _ _ _ _ Huc a E) in Hcu0.
  discriminate.
Qed.

Lemma cds_unique (id : code_id) (cd cd' : code_dep) :
  (id, cd) ∈ cds -> (id, cd') ∈ cds -> cd = cd'.
Proof.
  intros H H'.
  pose proof (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup H) as E.
  rewrite (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup H') in E.
  injection E as ->. reflexivity.
Qed.

(* The code of a code dep is bound in the unit, and its witness of known
   arity has the facts of the traversal. *)
Lemma cds_code (id : code_id) (cd : code_dep) :
  (id, cd) ∈ cds ->
  FlambdaReaperFlowProofs.known_witness_edges G id cd
  ∧ (∃ c, (id, c) ∈ unit_codes e0)
  ∧ id ∈ static_names code_ids_bound e0.
Proof.
  intros Hin.
  assert (Hid : id ∈ static_names code_ids_of_binding e0).
  { apply Hids. apply in_map_iff. exists (id, cd). auto. }
  destruct (proj1 unit_codes_names e0 id Hid) as [(c & Hc) Hb].
  split; [ | split; [ exists c; exact Hc | exact Hb ] ].
  pose proof (Hcode_edges id c Hc) as He.
  destruct c as [params my_closure ret exn metadata body].
  cbn [FlambdaReaperFlowProofs.code_edges] in He.
  destruct He as (cd' & Hin' & Hkw & _).
  rewrite (cds_unique id cd cd' Hin Hin'). exact Hkw.
Qed.

(* A result given to a node read off a witness that a call witness of
   known arity reaches: an alias, if the node has no source, or the
   result no usage. *)
Lemma link (x w s r : node) (f : field) :
  is_local f = false -> path T s w -> g_accessor G x f w ->
  g_constructor G s f r ->
  (s_any_source S x = false -> s_alias S x r = true)
  ∧ (s_any_usage S r = false -> s_alias S x r = true).
Proof.
  intros Hf Hp Hx Hr. split.
  - intros Hasx.
    assert (Hasw : s_any_source S w = false).
    { destruct (s_any_source S w) eqn:E; [ | reflexivity ].
      rewrite (any_source_accessor _ _ _ closedU w f x E Hx Hf) in Hasx.
      discriminate. }
    assert (Hass : s_any_source S s = false).
    { destruct (s_any_source S s) eqn:E; [ | reflexivity ].
      rewrite (any_source_fwd is_local F T Hleast s w Hp E) in Hasw.
      discriminate. }
    pose proof (sources_constructor _ _ _ closedU r f s Hr Hass) as Hss.
    destruct (sources_fwd is_local F T Hleast s w s Hp Hss) as [H | Hws];
      [ rewrite H in Hasw; discriminate | ].
    exact (alias_accessor_source _ _ _ closedU w s f x r Hx Hws Hasw Hr).
  - intros Haur.
    assert (Haus : s_any_usage S s = false).
    { destruct (s_any_usage S s) eqn:E; [ | reflexivity ].
      rewrite (any_usage_constructor _ _ _ closedU s f r E Hr Hf) in Haur.
      discriminate. }
    assert (Hauw : s_any_usage S w = false).
    { destruct (s_any_usage S w) eqn:E; [ | reflexivity ].
      rewrite (any_usage_back is_local F T Hleast s w Hp E) in Haus.
      discriminate. }
    pose proof (usages_accessor _ _ _ closedU x f w Hx Hauw) as Hww.
    destruct (usages_back is_local F T Hleast s w w Hp Hww) as [H | Hsw];
      [ rewrite H in Haus; discriminate | ].
    exact (alias_constructor_usage _ _ _ closedU s w f x r Hr Hsw Haus Hx).
Qed.

Definition dom_unbox (n : node) : Prop :=
  ∃ a, (a, n) ∈ dominated ∧ mem_node a to_unbox = true.

Lemma unboxed_iff (n : node) :
  unboxed_of_node unboxed n <> None <-> dom_unbox n.
Proof. exact (unboxed_of_unit F T dominated to_unbox e0 n). Qed.

(* The decisions [ds] of the rebuild for the continuations in scope, and
   the nodes [ce] of the traversal for their parameters. *)
Definition agree (ds : finite_map (list decision))
    (ce : finite_map (list node)) : Prop :=
  ∀ k ns, lookup k ce = Some ns ->
    (lookup k ds = None ∧ ∀ n, n ∈ ns -> s_any_usage S n = true)
    ∨ ∃ decs, lookup k ds = Some decs ∧ length decs = length ns
         ∧ ∀ i d n, nth_error decs i = Some d -> nth_error ns i = Some n ->
             decision_ok S unboxed d n.

Definition live (ctx : option code_id) : Prop :=
  ∀ c, ctx = Some c -> s_any_usage S (Node_code_id c) = true.

Definition call_ok (ads : list decision) (ap : apply_expr) : Prop :=
  ∀ i fd kind,
    ap_call_kind ap <> Indirect_unknown_arity ->
    nth_error (unit_callee_decisions F T cannot_change cds unboxed e0 ap) i
    = Some (fd, kind) ->
    arm_pair (apply_decision_at ads i kind) fd = true ->
    is_unbox_decision fd = false
    ∧ callee_result_has_source F T (node_of_simple (ap_callee ap)) i = false
    ∧ ∃ a, (a, known, node_of_simple (ap_callee ap)) ∈ f_accessor F.

Definition call_ok_indirect (ads : list decision) (ap : apply_expr) : Prop :=
  check_unknown_widgets F T = true -> check_unknown_witnesses F T = true ->
  ∀ i fd kind,
    ap_call_kind ap = Indirect_unknown_arity ->
    length (ap_args_arity ap) <= 1 ->
    nth_error (unit_callee_decisions F T cannot_change cds unboxed e0 ap) i
    = Some (fd, kind) ->
    arm_pair (apply_decision_at ads i kind) fd = true ->
    result_has_source_at F T unknown (node_of_simple (ap_callee ap)) i
    = false
    ∧ ∃ a, (a, unknown, node_of_simple (ap_callee ap)) ∈ f_accessor F.

Lemma apply_decision_facts (ds : finite_map (list decision))
    (ce : finite_map (list node)) (k : continuation) (ns : list node)
    (i : nat) (kind : flambda_kind) (x : node) :
  agree ds ce -> lookup k ce = Some ns -> nth_error ns i = Some x ->
  let ad := apply_decision_at
              (match lookup k ds with Some d => d | None => [] end) i kind in
  (is_unbox_decision ad = true -> dom_unbox x)
  ∧ (∀ kind', ad = Decided (Keep kind') -> s_has_usage S x = true)
  ∧ (dom_unbox x -> is_unbox_decision ad = true ∨ s_any_usage S x = true).
Proof.
  intros Hagree Hk Hx ad. unfold ad.
  destruct (Hagree k ns Hk) as [[Hn Hall] | (decs & Hd & Hlen & Hok)].
  - rewrite Hn. unfold apply_decision_at. rewrite nth_error_nil.
    pose proof (Hall x (nth_error_In _ _ Hx)) as Hau.
    split; [ discriminate | split; [ | intros _; right; exact Hau ] ].
    intros _ _. exact (has_usage_any _ _ _ closedU x Hau).
  - rewrite Hd. unfold apply_decision_at.
    destruct (nth_error decs i) as [d | ] eqn:Ed.
    + destruct (Hok i d x Ed Hx) as [Hu Hkeep].
      rewrite <- unboxed_iff, <- Hu. split; [ auto | split; [ exact Hkeep | ] ].
      intros H. left. exact H.
    + exfalso. apply nth_error_None in Ed.
      assert (Hlt : i < length ns) by (apply nth_error_Some; congruence).
      lia.
Qed.

Lemma arm_pair_cases (ad fd : decision) :
  arm_pair ad fd = true ->
  (is_unbox_decision ad = true ∧ is_unbox_decision fd = false)
  ∨ ((∃ kind, ad = Decided (Keep kind)) ∧ is_unbox_decision fd = true).
Proof.
  destruct ad as [[kind | ] | fields | kind], fd as [d | fields' | kind'];
    cbn; try discriminate; intros _; eauto.
Qed.

Lemma dominated_no_any_source (a x : node) :
  (a, x) ∈ dominated -> s_any_source S x = false.
Proof.
  intros H. apply (proj2 (mem_pair_In _ _ _)) in H.
  apply (dom_iff T dominated Hdom) in H. destruct H as (_ & _ & Hnm).
  destruct (s_any_source S x) eqn:E; [ | reflexivity ].
  exfalso. apply Hnm. left. exact E.
Qed.

(* A direct call: the decisions of its continuation and of its callee are
   the arm only where no result of the callee has a source. *)
Lemma apply_ok (ctx : option code_id) (ce : finite_map (list node))
    (ds : finite_map (list decision)) (ap : apply_expr) :
  FlambdaReaperFlowProofs.apply_edges G noted (unit_codes e0) ctx ce ap ->
  conts_counted (FlambdaReaperFlowProofs.counts ce) (Apply ap) = true ->
  live ctx -> agree ds ce ->
  call_ok (match lookup (ap_continuation ap) ds with
           | Some decisions => decisions
           | None => []
           end) ap.
Proof.
  intros (_ & _ & _ & returns & exn & Hret & _ & Hcall) Hcount Hlive Hagree
    i fd kind Hdirect Hfd Harm.
  cbn [conts_counted] in Hcount. apply andb_true_iff in Hcount as [Hcount _].
  unfold count_ok in Hcount. rewrite lookup_counts in Hcount.
  destruct (lookup (ap_continuation ap) ce) as [ns | ] eqn:Hk;
    cbn [option_map] in Hcount; [ | discriminate ].
  apply Nat.eqb_eq in Hcount.
  rewrite (Hret ns eq_refl) in Hcall.
  apply FlambdaReaperFlowProofs.known_call_edges_of in Hcall;
    [ | exact Hdirect ].
  destruct Hcall as (w & a & called & _ & Hres & _ & Hcalled & Hcalled_any
                     & Hcond & Hread).
  set (f := node_of_simple (ap_callee ap)) in *.
  assert (Hwa : s_alias S w a = true).
  { destruct ctx as [c | ]; cbn in Hcond.
    - exact (alias_propagate _ _ _ closedU _ _ _ (Hlive c eq_refl) Hcond).
    - exact (alias_input _ _ _ closedU _ _ Hcond). }
  pose proof Hfd as Hfd0.
  unfold unit_callee_decisions, callee_decisions in Hfd.
  set (kinds := unarize (ap_return_arity ap)) in *.
  assert (Hkind : nth_error kinds i = Some kind).
  { destruct (apply_change_u _ _ _ _ _ _ _ ap) as [ | pds rds].
    - rewrite nth_error_map in Hfd.
      destruct (nth_error kinds i) as [kind0 | ]; [ | discriminate ].
      cbn in Hfd. injection Hfd as _ ->. reflexivity.
    - apply nth_error_combine in Hfd as [_ H]. exact H. }
  assert (Hx : ∃ x, nth_error ns i = Some x).
  { destruct (nth_error ns i) as [x | ] eqn:E; [ exists x; reflexivity | ].
    apply nth_error_None in E.
    assert (Hlt : i < length kinds) by (apply nth_error_Some; congruence).
    lia. }
  destruct Hx as [x Hx].
  pose proof (Hres i x Hx) as Hxw.
  destruct (apply_decision_facts ds ce _ ns i kind x Hagree Hk Hx)
    as (Had_unbox & Had_keep & Had_dom).
  set (ad := apply_decision_at _ i kind) in *.
  pose proof (return_not_local is_local (Normal i) Hloc) as Hret_local.
  destruct (arm_pair_cases ad fd Harm)
    as [[Hadu Hfdu] | ((kk & Hadk) & Hfdu)].
  - (* [Unbox _, (Keep _ | Delete)]: the result has no source. *)
    split; [ exact Hfdu | ].
    split; [ | exists a; exact Hread ].
    destruct (Had_unbox Hadu) as (a0 & Ha0x & Ha0).
    pose proof (dominated_no_any_source a0 x Ha0x) as Hasx.
    assert (Hasw : s_any_source S w = false).
    { destruct (s_any_source S w) eqn:E; [ | reflexivity ].
      rewrite (any_source_accessor _ _ _ closedU w _ x E Hxw Hret_local)
        in Hasx.
      discriminate. }
    assert (Hasa : s_any_source S a = false).
    { destruct (s_any_source S a) eqn:E; [ | reflexivity ].
      rewrite (any_source_alias _ _ _ closedU a w Hwa E) in Hasw.
      discriminate. }
    destruct (callee_result_has_source F T f i) eqn:E;
      [ exfalso | reflexivity ].
    unfold callee_result_has_source, result_has_source_at in E.
    apply existsb_exists in E as ([[a1 f1] b1] & Ha1 & Ht).
    apply andb_true_iff in Ht as [Ht Hsrc].
    apply andb_true_iff in Ht as [Hf1 Hb1].
    apply field_eqb_eq in Hf1. apply node_eqb_eq in Hb1. subst f1 b1.
    destruct (widgets_same F T a1 a f Hwidgets Ha1 Hread) as [Hany1 Hsrc1].
    apply orb_true_iff in Hsrc as [Hsrc | Hsrc].
    { cbn in Hany1. rewrite Hsrc in Hany1. cbn in Hasa.
      rewrite <- Hany1 in Hasa.
      discriminate. }
    apply existsb_exists in Hsrc as ([a2 s] & Hs & Ht).
    apply andb_true_iff in Ht as [Ha2 Hc].
    apply node_eqb_eq in Ha2. subst a2.
    apply existsb_exists in Hc as ([[s' f'] r] & Hc & Ht).
    apply andb_true_iff in Ht as [Ht Hhr].
    apply andb_true_iff in Ht as [Hs' Hf'].
    apply node_eqb_eq in Hs'. apply field_eqb_eq in Hf'. subst s' f'.
    apply (mem_pair_In _ _ _) in Hs.
    pose proof (Hsrc1 s Hs) as Has.
    assert (Hhs : s_has_source S s = true).
    { exact (has_source_constructor _ _ _ closedU r _ s Hhr Hc). }
    destruct (kw_source F T cds Hknown a f s Hread Has Hhs)
      as (id_s & cd_s & Hcd_s & ->).
    destruct (cds_code id_s cd_s Hcd_s) as (Hkw & _ & Hbound).
    destruct Hkw as (_ & Hkw_res & _ & Hkw_id & _).
    pose proof (path_snoc T _ _ _ (sources_path is_local F T Hleast a _ Has)
                  Hwa) as Hpath.
    pose proof (proj1 (link _ _ _ _ _ Hret_local Hpath Hxw Hc) Hasx) as Hxr.
    pose proof (flow_unboxed is_local F T Hleast dominated Hdom x r a0 Hxr
                  (proj2 (mem_pair_In _ _ _) Ha0x) Hhr) as Ha0r.
    pose proof (proj1 (kw_constructor F T cds Hknown id_s cd_s _ r Hcd_s Hc) i
                  eq_refl) as Hnth_r.
    assert (Hcannot : mem_node (Node_code_id id_s) cannot_change = false).
    { exact (flow_cannot_change is_local F T Hleast cannot_change
               max_unbox_size lambda_lifting dominated to_unbox Hdom Hunbox x r
               a0 _ _ _ Hxr (proj2 (mem_pair_In _ _ _) Ha0x) Ha0 Hhr Hc
               eq_refl Hkw_id). }
    unfold apply_change_u in Hfd.
    destruct (code_id_actually_called F S ap) as [id1 | ] eqn:Ecall.
    + destruct (called_some F S ap id1 Ecall) as (ids & Eids & Esing).
      pose proof (directly_member F S f ids a _ id_s Eids Hread Has Hhs Hkw_id)
        as Hin_s.
      pose proof (singleton_some ids id1 id_s Esing Hin_s) as <-.
      destruct (lookup id_s (bound_code_metadata e0)) as [md | ] eqn:Emd.
      * unfold code_change_u in Hfd.
        destruct (cannot_change_calling_convention
                    (fun n => mem_node n cannot_change)
                    (static_names code_ids_bound e0) id_s) eqn:Ecc.
        -- unfold cannot_change_calling_convention in Ecc.
           apply orb_true_iff in Ecc as [Ecc | Ecc].
           ++ rewrite (proj2 (FlambdaMapProofs.among_In _ _) Hbound) in Ecc.
              discriminate.
           ++ rewrite Ecc in Hcannot. discriminate.
        -- destruct (lookup id_s cds) as [cd | ] eqn:Ecd.
           ++ pose proof (FlambdaMapProofs.lookup_in _ _ _ Ecd) as Hcd.
              rewrite (cds_unique id_s _ _ Hcd Hcd_s) in Hfd.
              apply nth_error_combine in Hfd as [Hfd _].
              destruct (decisions_u_unbox _ _ _ _ _ _ Hfd)
                as (n & Hn & Hiff).
              rewrite Hnth_r in Hn. injection Hn as <-.
              rewrite Hfdu in Hiff.
              assert (Hdr : dom_unbox r).
              { exists a0. split; [ | exact Ha0 ].
                apply (mem_pair_In _ _ _). exact Ha0r. }
              discriminate (proj2 Hiff (proj2 (unboxed_iff r) Hdr)).
           ++ rewrite (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup
                         Hcd_s) in Ecd.
              discriminate.
      * destruct (cds_code id_s cd_s Hcd_s) as (_ & (c & Hc_s) & _).
        apply FlambdaMapProofs.lookup_none_not_in in Emd. apply Emd.
        rewrite (proj1 bound_code_metadata_codes e0), map_map.
        apply in_map_iff. exists (id_s, c). split; [ reflexivity | exact Hc_s ].
    + destruct (called_none F S ap Hdirect Ecall)
        as [(c0 & Hc0) | [Enone | (ids & Eids & Esing)]].
      * unfold f in Hread. rewrite Hc0 in Hread. cbn [node_of_simple] in Hread.
        pose proof (any_source_input _ _ _ closedU _ Hconsts) as Hac.
        rewrite (any_source_accessor _ _ _ closedU _ _ a Hac Hread
                   (known_not_local is_local Hloc))
          in Hasa.
        discriminate.
      * destruct (directly_none F S f Enone) as (a3 & Ha3 & Hany3).
        destruct (widgets_same F T a3 a f Hwidgets Ha3 Hread) as [Heq _].
        cbn in Hany3, Heq, Hasa. rewrite Hany3 in Heq. rewrite <- Heq in Hasa.
        discriminate.
      * pose proof (directly_member F S f ids a _ id_s Eids Hread Has Hhs
                      Hkw_id) as Hin_s.
        destruct (singleton_none ids id_s Esing Hin_s) as (id2 & Hid2 & Hne).
        destruct (directly_inv F S f ids id2 Eids Hid2)
          as (a4 & s2 & Ha4 & Hs2 & Hhs2 & Hc2).
        pose proof (proj2 (widgets_same F T a4 a f Hwidgets Ha4 Hread) s2 Hs2)
          as Has2.
        assert (Hne_s : s2 <> cd_known_arity_call_witness cd_s).
        { intros ->.
          pose proof (proj2 (kw_constructor F T cds Hknown id_s cd_s _ _ Hcd_s
                               Hc2) eq_refl) as Heq.
          injection Heq as Heq. apply Hne. exact Heq. }
        assert (Hhu_w : s_has_usage S w = true).
        { pose proof (has_usage_any _ _ _ closedU _
                        (any_usage_input _ _ _ closedU _ Hcalled_any)) as H.
          exact (has_usage_accessor _ _ _ closedU _ _ w H Hcalled). }
        assert (Hhu_s : s_has_usage S (cd_known_arity_call_witness cd_s)
                        = true)
          by exact (has_usage_back _ _ Hpath Hhu_w).
        assert (Hw : cannot_change_witness_calling_convention F T
                       (cd_known_arity_call_witness cd_s) = true).
        { unfold cannot_change_witness_calling_convention.
          rewrite !orb_true_iff.
          destruct (s_any_usage S (cd_known_arity_call_witness cd_s)) eqn:Eau.
          - left. left. left. left. left. exact Eau.
          - left. left. left. left. right.
            assert (Hauw : s_any_usage S w = false).
            { destruct (s_any_usage S w) eqn:E; [ | reflexivity ].
              rewrite (any_usage_back is_local F T Hleast _ _ Hpath E) in Eau.
              discriminate. }
            pose proof (usages_accessor _ _ _ closedU _ _ w Hcalled Hauw)
              as Hww.
            destruct (usages_back is_local F T Hleast _ _ _ Hpath Hww)
              as [H | Hsw]; [ rewrite H in Eau; discriminate | ].
            pose proof (sources_alias _ _ _ closedU a w s2 Has2 Hasa Hwa
                          Hasw) as Hws2.
            unfold witness_other_source. apply existsb_exists.
            exists (cd_known_arity_call_witness cd_s, w).
            split; [ apply (mem_pair_In _ _ _); exact Hsw | ].
            rewrite node_eqb_refl. cbn [andb].
            apply andb_true_iff. split.
            + apply existsb_exists. exists (w, s2).
              split; [ apply (mem_pair_In _ _ _); exact Hws2 | ].
              rewrite node_eqb_refl. cbn in Hhs2. rewrite Hhs2.
              rewrite FlambdaReaperUnboxingProofs.node_eqb_false
                by (intros E; apply Hne_s; exact E).
              reflexivity.
            + unfold code_id_read_with_usage. apply existsb_exists.
              exists (called, Field_code_id_of_call_witness, w).
              split; [ exact Hcalled | ].
              rewrite field_eqb_refl, node_eqb_refl. cbn [andb].
              exact (has_usage_any _ _ _ closedU _
                       (any_usage_input _ _ _ closedU _ Hcalled_any)). }
        rewrite (cc_closed_witness F T cannot_change _ _ Hcc Hkw_id Hhu_s Hw)
          in Hcannot.
        discriminate.
  - (* [Keep _, Unbox _]: never. *)
    exfalso.
    unfold apply_change_u in Hfd.
    destruct (code_id_actually_called F S ap) as [id1 | ] eqn:Ecall.
    2:{ rewrite nth_error_map in Hfd.
        destruct (nth_error kinds i); [ | discriminate ].
        cbn in Hfd. injection Hfd as <- _. discriminate Hfdu. }
    destruct (lookup id1 (bound_code_metadata e0)) as [md | ] eqn:Emd.
    2:{ rewrite nth_error_map in Hfd.
        destruct (nth_error kinds i); [ | discriminate ].
        cbn in Hfd. injection Hfd as <- _. discriminate Hfdu. }
    unfold code_change_u in Hfd.
    destruct (cannot_change_calling_convention
                (fun n => mem_node n cannot_change)
                (static_names code_ids_bound e0) id1).
    { rewrite nth_error_map in Hfd.
      destruct (nth_error kinds i); [ | discriminate ].
      cbn in Hfd. injection Hfd as <- _. discriminate Hfdu. }
    destruct (lookup id1 cds) as [cd | ] eqn:Ecd.
    2:{ rewrite nth_error_map in Hfd.
        destruct (nth_error kinds i); [ | discriminate ].
        cbn in Hfd. injection Hfd as <- _. discriminate Hfdu. }
    apply nth_error_combine in Hfd as [Hfd _].
    destruct (decisions_u_unbox _ _ _ _ _ _ Hfd) as (r & Hr & Hiff).
    pose proof (proj1 Hiff Hfdu) as Hdr.
    apply unboxed_iff in Hdr. destruct Hdr as (a0 & Ha0r & Ha0).
    pose proof (to_unbox_any_usage a0 Ha0) as Hau0.
    apply (proj2 (mem_pair_In _ _ _)) in Ha0r as Ha0r'.
    apply (dom_iff T dominated Hdom) in Ha0r'. destruct Ha0r' as (Hsr & _ & _).
    pose proof (sources_path is_local F T Hleast r a0 Hsr) as Hpa0r.
    assert (Haur : s_any_usage S r = false).
    { destruct (s_any_usage S r) eqn:E; [ | reflexivity ].
      rewrite (any_usage_back is_local F T Hleast _ _ Hpa0r E) in Hau0.
      discriminate. }
    destruct (called_some F S ap id1 Ecall) as (ids & Eids & Esing).
    assert (Hid1 : id1 ∈ ids).
    { pose proof (proj1 (FlambdaMapProofs.keyed_get_singleton _ Pos.eqb
                           pos_eqb_decides id1 ids) Esing id1) as H.
      rewrite Pos.eqb_refl in H.
      exact (proj1 (FlambdaMapProofs.keyed_among_in _ Pos.eqb pos_eqb_decides
                      id1 ids) H). }
    destruct (directly_inv F S f ids id1 Eids Hid1)
      as (a1 & s & Ha1 & Hs1 & Hhs & Hcid).
    destruct (kw_source F T cds Hknown a1 f s Ha1 Hs1 Hhs)
      as (id' & cd' & Hcd' & ->).
    pose proof (proj2 (kw_constructor F T cds Hknown id' cd' _ _ Hcd' Hcid)
                  eq_refl) as Heq.
    injection Heq as <-.
    pose proof (FlambdaMapProofs.lookup_in _ _ _ Ecd) as Hcd.
    rewrite (cds_unique id1 _ _ Hcd' Hcd) in *.
    pose proof (proj2 (widgets_same F T a1 a f Hwidgets Ha1 Hread) _ Hs1)
      as Has.
    pose proof (path_snoc T _ _ _ (sources_path is_local F T Hleast a _ Has)
                  Hwa) as Hpath.
    destruct (cds_code id1 cd Hcd) as (Hkw & _ & _).
    destruct Hkw as (_ & Hkw_res & _).
    pose proof (Hkw_res i r Hr) as Hc.
    pose proof (proj2 (link _ _ _ _ _ Hret_local Hpath Hxw Hc) Haur) as Hxr.
    pose proof (flow_kept is_local F T Hleast cannot_change max_unbox_size
                  lambda_lifting dominated to_unbox Hdom Hunbox x r a0 Hargs
                  Hxr (proj2 (mem_pair_In _ _ _) Ha0r) Ha0
                  (Had_keep kk Hadk)) as Ha0x.
    destruct (Had_dom (ex_intro _ a0 (conj (proj1 (mem_pair_In _ _ _) Ha0x)
                                          Ha0)))
      as [H | H].
    + rewrite Hadk in H. discriminate.
    + rewrite (any_usage_alias _ _ _ closedU x r H Hxr) in Haur. discriminate.
Qed.

(* A call of unknown arity with one group of arguments: its callee keeps
   every result, and the parameter of its continuation is unboxed only
   where no result of the callee has a source. *)
Lemma apply_ok_indirect (ctx : option code_id) (ce : finite_map (list node))
    (ds : finite_map (list decision)) (ap : apply_expr) :
  FlambdaReaperFlowProofs.apply_edges G noted (unit_codes e0) ctx ce ap ->
  conts_counted (FlambdaReaperFlowProofs.counts ce) (Apply ap) = true ->
  live ctx -> agree ds ce ->
  call_ok_indirect (match lookup (ap_continuation ap) ds with
                    | Some decisions => decisions
                    | None => []
                    end) ap.
Proof.
  intros (_ & _ & _ & returns & exn & Hret & _ & Hcall) Hcount Hlive Hagree
    Huw Hux i fd kind Hind Hlen Hfd Harm.
  cbn [conts_counted] in Hcount. apply andb_true_iff in Hcount as [Hcount _].
  unfold count_ok in Hcount. rewrite lookup_counts in Hcount.
  destruct (lookup (ap_continuation ap) ce) as [ns | ] eqn:Hk;
    cbn [option_map] in Hcount; [ | discriminate ].
  apply Nat.eqb_eq in Hcount.
  rewrite (Hret ns eq_refl) in Hcall.
  unfold FlambdaReaperFlowProofs.call_edges in Hcall. rewrite Hind in Hcall.
  destruct Hcall as (ws & a & called & Hws & Hchain & Hcalled_any & Hcond
                     & Hread).
  pose proof (one_group ap Hlen) as Hg.
  destruct (FlambdaReaperFlowProofs.call_groups ap) as [ | g [ | g' gs]];
    cbn in Hg; try discriminate.
  destruct ws as [ | w [ | w' ws']]; cbn in Hws; try discriminate.
  cbn [combine FlambdaReaperFlowProofs.caller_chain_edges] in Hchain.
  destruct Hchain as (_ & _ & Hcalled & Hres & _).
  specialize (Hcond w eq_refl).
  set (f := node_of_simple (ap_callee ap)) in *.
  assert (Hwa : s_alias S w a = true).
  { destruct ctx as [c | ]; cbn in Hcond.
    - exact (alias_propagate _ _ _ closedU _ _ _ (Hlive c eq_refl) Hcond).
    - exact (alias_input _ _ _ closedU _ _ Hcond). }
  unfold unit_callee_decisions, callee_decisions, apply_change_u in Hfd.
  unfold code_id_actually_called in Hfd. rewrite Hind in Hfd.
  set (kinds := unarize (ap_return_arity ap)) in *.
  rewrite nth_error_map in Hfd.
  destruct (nth_error kinds i) as [kind0 | ] eqn:Hkind; [ | discriminate ].
  cbn in Hfd. injection Hfd as <- <-.
  assert (Hx : ∃ x, nth_error ns i = Some x).
  { destruct (nth_error ns i) as [x | ] eqn:E; [ exists x; reflexivity | ].
    apply nth_error_None in E.
    assert (Hlt : i < length kinds) by (apply nth_error_Some; congruence).
    lia. }
  destruct Hx as [x Hx].
  pose proof (Hres i x Hx) as Hxw.
  destruct (apply_decision_facts ds ce _ ns i kind0 x Hagree Hk Hx)
    as (Had_unbox & _ & _).
  set (ad := apply_decision_at _ i kind0) in *.
  pose proof (return_not_local is_local (Normal i) Hloc) as Hret_local.
  destruct (arm_pair_cases ad _ Harm) as [[Hadu _] | (_ & Hfdu)];
    [ | discriminate Hfdu ].
  split; [ | exists a; exact Hread ].
  destruct (Had_unbox Hadu) as (a0 & Ha0x & Ha0).
  pose proof (dominated_no_any_source a0 x Ha0x) as Hasx.
  assert (Hasw : s_any_source S w = false).
  { destruct (s_any_source S w) eqn:E; [ | reflexivity ].
    rewrite (any_source_accessor _ _ _ closedU w _ x E Hxw Hret_local)
      in Hasx.
    discriminate. }
  assert (Hasa : s_any_source S a = false).
  { destruct (s_any_source S a) eqn:E; [ | reflexivity ].
    rewrite (any_source_alias _ _ _ closedU a w Hwa E) in Hasw.
    discriminate. }
  destruct (result_has_source_at F T unknown f i) eqn:E;
    [ exfalso | reflexivity ].
  unfold result_has_source_at in E.
  apply existsb_exists in E as ([[a1 f1] b1] & Ha1 & Ht).
  apply andb_true_iff in Ht as [Ht Hsrc].
  apply andb_true_iff in Ht as [Hf1 Hb1].
  apply field_eqb_eq in Hf1. apply node_eqb_eq in Hb1. subst f1 b1.
  destruct (unknown_widgets_same F T a1 a f Huw Ha1 Hread) as [Hany1 Hsrc1].
  apply orb_true_iff in Hsrc as [Hsrc | Hsrc].
  { cbn in Hany1. rewrite Hsrc in Hany1. cbn in Hasa.
    rewrite <- Hany1 in Hasa. discriminate. }
  apply existsb_exists in Hsrc as ([a2 s] & Hs & Ht).
  apply andb_true_iff in Ht as [Ha2 Hc].
  apply node_eqb_eq in Ha2. subst a2.
  apply existsb_exists in Hc as ([[s' f'] r] & Hc & Ht).
  apply andb_true_iff in Ht as [Ht Hhr].
  apply andb_true_iff in Ht as [Hs' Hf'].
  apply node_eqb_eq in Hs'. apply field_eqb_eq in Hf'. subst s' f'.
  apply (mem_pair_In _ _ _) in Hs.
  pose proof (Hsrc1 s Hs) as Has.
  assert (Hhs : s_has_source S s = true).
  { exact (has_source_constructor _ _ _ closedU r _ s Hhr Hc). }
  pose proof (path_snoc T _ _ _ (sources_path is_local F T Hleast a _ Has)
                Hwa) as Hpath.
  pose proof (proj1 (link _ _ _ _ _ Hret_local Hpath Hxw Hc) Hasx) as Hxr.
  destruct (unknown_source F T a f s Hux Hread Has Hhs)
    as [(b & Hb) (codeid & Hcid)].
  assert (Hhu_w : s_has_usage S w = true).
  { pose proof (has_usage_any _ _ _ closedU _
                  (any_usage_input _ _ _ closedU _ Hcalled_any)) as H.
    exact (has_usage_accessor _ _ _ closedU _ _ w H Hcalled). }
  pose proof (has_usage_back _ _ Hpath Hhu_w) as Hhu_s.
  pose proof (cc_closed_unknown F T cannot_change b s codeid Hcc Hb Hhu_s
                Hcid) as Hin_cc.
  rewrite (flow_cannot_change is_local F T Hleast cannot_change
             max_unbox_size lambda_lifting dominated to_unbox Hdom Hunbox x r
             a0 _ _ _ Hxr (proj2 (mem_pair_In _ _ _) Ha0x) Ha0 Hhr Hc
             eq_refl Hcid)
    in Hin_cc.
  discriminate.
Qed.

Lemma agree_cons (ds : finite_map (list decision))
    (ce : finite_map (list node)) (k : continuation) (decs : list decision)
    (ns : list node) :
  agree ds ce -> length decs = length ns ->
  (∀ i d n, nth_error decs i = Some d -> nth_error ns i = Some n ->
     decision_ok S unboxed d n) ->
  agree ((k, decs) :: ds) ((k, ns) :: ce).
Proof.
  intros Hagree Hlen Hok k' ns' H. cbn [lookup Keyed.lookup] in *.
  destruct (Pos.eqb k' k); [ | exact (Hagree k' ns' H) ].
  injection H as <-. right. exists decs. auto.
Qed.

Lemma lookup_map_pairs {A B : Type} (g : A -> B) (l : list (positive * A))
    (k : positive) :
  lookup k (map (fun '(k', a) => (k', g a)) l) = option_map g (lookup k l).
Proof.
  induction l as [ | [k' a] l IH]; [ reflexivity | ].
  cbn [map lookup Keyed.lookup]. destruct (Pos.eqb k k'); [ reflexivity | ].
  exact IH.
Qed.

Lemma agree_app (ds : finite_map (list decision))
    (ce : finite_map (list node))
    (handlers : list (continuation * continuation_handler))
    (D : continuation_handler -> list decision)
    (N : continuation_handler -> list node) :
  agree ds ce ->
  (∀ k h, (k, h) ∈ handlers ->
     length (D h) = length (N h)
     ∧ ∀ i d n, nth_error (D h) i = Some d -> nth_error (N h) i = Some n ->
         decision_ok S unboxed d n) ->
  agree (map (fun '(k, h) => (k, D h)) handlers ++ ds)
    (map (fun '(k, h) => (k, N h)) handlers ++ ce).
Proof.
  intros Hagree Hh k ns H.
  rewrite FlambdaMapProofs.lookup_app, lookup_map_pairs in H.
  rewrite FlambdaMapProofs.lookup_app, lookup_map_pairs.
  destruct (lookup k handlers) as [h | ] eqn:E; cbn [option_map] in *.
  - injection H as <-. right. exists (D h). split; [ reflexivity | ].
    exact (Hh k h (FlambdaMapProofs.lookup_in _ _ _ E)).
  - exact (Hagree k ns H).
Qed.

Lemma handler_agree (is_exn : bool) (h : continuation_handler) :
  handler_plain_kinds h = true ->
  let decs := handler_decisions true S unboxed is_exn h in
  length decs = length (FlambdaReaperTraversal.handler_params h)
  ∧ ∀ i d n, nth_error decs i = Some d ->
      nth_error (FlambdaReaperTraversal.handler_params h) i = Some n ->
      decision_ok S unboxed d n.
Proof.
  destruct h as [params metadata body]. cbn [handler_plain_kinds].
  intros Hk. apply andb_true_iff in Hk as [Hk _].
  destruct (handler_decisions_ok S unboxed is_exn params metadata body Hk)
    as [Hlen Hok].
  cbn [FlambdaReaperTraversal.handler_params].
  split; [ rewrite length_map; exact Hlen | ].
  intros i d n Hd Hn. rewrite nth_error_map in Hn.
  destruct (nth_error params i) as [x | ] eqn:Ex; [ | discriminate ].
  cbn in Hn. injection Hn as <-. exact (Hok i d x Hd Ex).
Qed.

Lemma decisions_length (fixed_arity : bool) (nodes : list node)
    (kinds : list flambda_kind) :
  length nodes = length kinds ->
  length (if fixed_arity then fixed_arity_decisions S unboxed nodes kinds
          else decisions_u S unboxed nodes kinds)
  = length nodes.
Proof.
  intros H. destruct fixed_arity; unfold fixed_arity_decisions, decisions_u;
    rewrite length_map, length_combine, H; apply Nat.min_id.
Qed.

(* The continuations of code: the decisions of the rebuild and the nodes
   of the traversal. *)
Lemma code_agree (id : code_id) (cd : code_dep) (metadata : code_metadata)
    (ret exn : continuation) :
  lookup id cds = Some cd -> ret <> exn ->
  length (cd_return cd) = length (unarize (cm_result_arity metadata)) ->
  forallb plain_kind (unarize (cm_result_arity metadata)) = true ->
  agree
    (code_continuation_decisions S cds unboxed id metadata
       (code_change_u S (fun n => mem_node n cannot_change) cds
          (static_names code_ids_bound e0) unboxed id metadata)
       ret exn)
    [(exn, [cd_exn cd]); (ret, cd_return cd)].
Proof.
  intros Hcd Hne Hlen Hplain.
  set (kinds := unarize (cm_result_arity metadata)) in *.
  set (exn_decs := fixed_arity_decisions S unboxed [cd_exn cd] [Value]).
  assert (Hexn : length exn_decs = length [cd_exn cd]
                 ∧ ∀ i d n, nth_error exn_decs i = Some d ->
                     nth_error [cd_exn cd] i = Some n ->
                     decision_ok S unboxed d n).
  { split; [ exact (decisions_length true [cd_exn cd] [Value] eq_refl) | ].
    exact (node_decisions_ok S unboxed true [cd_exn cd] [Value] eq_refl). }
  assert (Hret : ∀ fixed_arity : bool,
            let decs := if fixed_arity
                        then fixed_arity_decisions S unboxed (cd_return cd)
                               kinds
                        else decisions_u S unboxed (cd_return cd) kinds in
            length decs = length (cd_return cd)
            ∧ ∀ i d n, nth_error decs i = Some d ->
                nth_error (cd_return cd) i = Some n ->
                decision_ok S unboxed d n).
  { intros fixed_arity decs.
    split; [ exact (decisions_length fixed_arity _ _ Hlen) | ].
    exact (node_decisions_ok S unboxed fixed_arity _ _ Hplain). }
  assert (Hgen : ∀ decs : list decision,
            length decs = length (cd_return cd)
            ∧ (∀ i d n, nth_error decs i = Some d ->
                 nth_error (cd_return cd) i = Some n ->
                 decision_ok S unboxed d n) ->
            agree ([(ret, decs)] ++ [(exn, exn_decs)])
              [(exn, [cd_exn cd]); (ret, cd_return cd)]).
  { intros decs [Hl Hok] k ns H. cbn [app lookup Keyed.lookup] in *.
    destruct (Pos.eqb k exn) eqn:Ekx.
    - apply Pos.eqb_eq in Ekx. subst k. injection H as <-.
      replace (Pos.eqb exn ret) with false
        by (symmetry; apply Pos.eqb_neq; intros E; apply Hne; symmetry;
            exact E).
      right. exists exn_decs.
      split; [ reflexivity | exact Hexn ].
    - destruct (Pos.eqb k ret) eqn:Ekr; [ | discriminate ].
      injection H as <-. right. exists decs. auto. }
  unfold code_continuation_decisions. rewrite Hcd.
  unfold code_change_u.
  destruct (cannot_change_calling_convention _ _ id).
  - exact (Hgen _ (Hret true)).
  - rewrite Hcd. exact (Hgen _ (Hret false)).
Qed.

Lemma used_code_live (id : code_id) :
  is_code_id_used S id = true -> live (Some id).
Proof.
  intros H c [= <-]. unfold check_used_code_ids in Hcodes.
  rewrite forallb_forall in Hcodes.
  exact (Hcodes _ (proj1 (mem_node_In _ _) H)).
Qed.

Lemma counts_rec (ce : finite_map (list node))
    (handlers : list (continuation * continuation_handler)) :
  map (fun '(k, h) => (k, length (FlambdaReaperTraversal.handler_params h)))
    handlers
  ++ FlambdaReaperFlowProofs.counts ce
  = FlambdaReaperFlowProofs.counts
      (FlambdaReaperFlowProofs.rec_conts ce handlers).
Proof.
  unfold FlambdaReaperFlowProofs.counts, FlambdaReaperFlowProofs.rec_conts.
  rewrite map_app, map_map. f_equal. apply map_ext. intros [k h].
  reflexivity.
Qed.

Abbreviation edges :=
  (FlambdaReaperFlowProofs.expr_edges G noted deps cds (unit_kept e0)
     (unit_codes e0) (static_names symbols_of_binding e0) slots_read False).
Abbreviation handler_edges :=
  (FlambdaReaperFlowProofs.handler_edges G noted deps cds (unit_kept e0)
     (unit_codes e0) (static_names symbols_of_binding e0) slots_read False).
Abbreviation calls :=
  (rebuilt_calls S (fun n => mem_node n cannot_change) cds
     (static_names code_ids_bound e0) unboxed).
Abbreviation handler_calls :=
  (handler_rebuilt_calls S (fun n => mem_node n cannot_change) cds
     (static_names code_ids_bound e0) unboxed).

(* Every call that the rebuild makes, in the unit and in code, is
   [call_ok]. *)
Lemma calls_ok :
  (∀ e : expr, ∀ ctx ce ds,
     edges ctx ce e ->
     conts_counted (FlambdaReaperFlowProofs.counts ce) e = true ->
     plain_kinds e = true -> live ctx -> agree ds ce ->
     ∀ ads ap, (ads, ap) ∈ calls ds e ->
       call_ok ads ap ∧ call_ok_indirect ads ap)
  ∧ (∀ h : continuation_handler, ∀ ctx ce ds,
       handler_edges ctx ce h ->
       handler_conts_counted (FlambdaReaperFlowProofs.counts ce) h = true ->
       handler_plain_kinds h = true -> live ctx -> agree ds ce ->
       ∀ ads ap, (ads, ap) ∈ handler_calls ds h ->
       call_ok ads ap ∧ call_ok_indirect ads ap).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH ctx ce ds [_ He] Hc Hk Hl Ha ads ap Hin.
    exact (IH ctx ce ds He Hc Hk Hl Ha ads ap Hin).
  - intros bindings body IHc IH ctx ce ds [Hb He] Hc Hk Hl Ha ads ap Hin.
    cbn [plain_kinds] in Hk. apply andb_true_iff in Hk as [Hkb Hk].
    cbn [rebuilt_calls] in Hin. apply in_app_or in Hin as [Hin | Hin];
      [ | exact (IH ctx ce ds He Hc Hk Hl Ha ads ap Hin) ].
    apply in_flat_map in Hin as (b & Hbin & Hin).
    destruct b as [id c | id | sym const]; cbn [binding_rebuilt_calls] in Hin;
      [ | destruct Hin | destruct Hin ].
    destruct (is_code_id_used S id) eqn:Eused; [ | destruct Hin ].
    pose proof (proj1 (FlambdaMapProofs.every_in _ _ _) Hb _ Hbin) as Hbe.
    cbn [FlambdaReaperFlowProofs.binding_edges] in Hbe.
    destruct Hbe as [_ Hce].
    rewrite forallb_forall in Hkb. pose proof (Hkb _ Hbin) as Hkc.
    pose proof (IHc id c Hbin) as IHbody.
    destruct c as [params my_closure ret exn metadata cbody].
    cbn [code_rebuilt_calls] in Hin. cbn [code_body] in IHbody.
    cbn [binding_plain_kinds code_plain_kinds] in Hkc.
    apply andb_true_iff in Hkc as [Hkr Hkc].
    cbn [FlambdaReaperFlowProofs.code_edges] in Hce.
    destruct Hce as (cd & Hcd & _ & _ & _ & _ & Hlen & _ & _ & Hne & _ & Hcc'
                     & Hce).
    pose proof (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd)
      as Hlookup.
    refine (IHbody (Some id) _ _ Hce _ Hkc (used_code_live id Eused)
              (code_agree id cd metadata ret exn Hlookup Hne Hlen Hkr) ads ap
              Hin).
    unfold FlambdaReaperFlowProofs.counts. cbn [map length]. rewrite Hlen.
    exact Hcc'.
  - intros k h body IHh IH ctx ce ds [Hh He] Hc Hk Hl Ha ads ap Hin.
    cbn [conts_counted] in Hc. apply andb_true_iff in Hc as [Hch Hc].
    cbn [plain_kinds] in Hk. apply andb_true_iff in Hk as [Hkh Hk].
    cbn [rebuilt_calls] in Hin. apply in_app_or in Hin as [Hin | Hin];
      [ exact (IHh ctx ce ds Hh Hch Hkh Hl Ha ads ap Hin) | ].
    destruct (handler_agree (handler_is_exn_handler h) h Hkh) as [Hlen Hok].
    exact (IH ctx ((k, FlambdaReaperTraversal.handler_params h) :: ce) _ He Hc
             Hk Hl (agree_cons ds ce k _ _ Ha Hlen Hok) ads ap Hin).
  - intros handlers body IHh IH ctx ce ds [Hhs He] Hc Hk Hl Ha ads ap Hin.
    cbn [conts_counted] in Hc. apply andb_true_iff in Hc as [Hch Hc].
    rewrite counts_rec in Hch, Hc.
    cbn [plain_kinds] in Hk. apply andb_true_iff in Hk as [Hkh Hk].
    rewrite forallb_forall in Hch, Hkh.
    assert (Ha' : agree
                    (map (fun '(k, h) =>
                            (k, handler_decisions true S unboxed false h))
                       handlers
                     ++ ds)
                    (FlambdaReaperFlowProofs.rec_conts ce handlers)).
    { apply agree_app; [ exact Ha | ]. intros k h Hkh'.
      exact (handler_agree false h (Hkh _ Hkh')). }
    cbn [rebuilt_calls] in Hin. apply in_app_or in Hin as [Hin | Hin];
      [ | exact (IH ctx _ _ He Hc Hk Hl Ha' ads ap Hin) ].
    apply in_flat_map in Hin as ([k h] & Hkh' & Hin).
    exact (IHh k h Hkh' ctx _ _
             (proj1 (FlambdaMapProofs.every_in _ _ _) Hhs _ Hkh')
             (Hch _ Hkh') (Hkh _ Hkh') Hl Ha' ads ap Hin).
  - intros ap ctx ce ds He Hc _ Hl Ha ads ap' Hin.
    cbn [rebuilt_calls] in Hin. destruct Hin as [[= <- <-] | []].
    exact (conj (apply_ok ctx ce ds ap He Hc Hl Ha)
             (apply_ok_indirect ctx ce ds ap He Hc Hl Ha)).
  - intros ac ctx ce ds _ _ _ _ _ ads ap [].
  - intros scrutinee arms ctx ce ds _ _ _ _ _ ads ap [].
  - intros ctx ce ds _ _ _ _ _ ads ap [].
  - intros params metadata handler IH ctx ce ds He Hc Hk Hl Ha ads ap Hin.
    cbn [handler_plain_kinds] in Hk. apply andb_true_iff in Hk as [_ Hk].
    exact (IH ctx ce ds He Hc Hk Hl Ha ads ap Hin).
Qed.


(* ------------------------------------------------------------------------ *)
(* The call of the arm never returns                                        *)
(* ------------------------------------------------------------------------ *)

(* A result with no source at the node [a] that a call reads off its
   callee: [a] has no source of any source, and no source of [a] holds
   a result of that rank with a source. *)
Lemma no_source_at (fld : field) (f a s r : node) (i : nat) :
  result_has_source_at F T fld f i = false -> (a, fld, f) ∈ f_accessor F ->
  s_any_source S a = false
  ∧ (s_sources S a s = true ->
     (s, Field_return_of_call (Normal i), r) ∈ f_constructor F ->
     s_has_source S r = false).
Proof.
  intros H Ha. unfold result_has_source_at in H.
  assert (Hrow : ∀ b : bool,
            (field_eqb fld fld && node_eqb f f
             && (mem_node a (t_any_source T)
                 || existsb
                      (fun '(a', s0) =>
                         node_eqb a' a
                         && existsb
                              (fun '(s', f', r0) =>
                                 node_eqb s' s0
                                 && field_eqb f'
                                      (Field_return_of_call (Normal i))
                                 && mem_node r0 (t_has_source T))
                              (f_constructor F))
                      (t_sources T))) = true -> b = false).
  { intros b E. exfalso. rewrite <- (Bool.not_true_iff_false _) in H.
    apply H. apply existsb_exists. exists (a, fld, f).
    split; [ exact Ha | exact E ]. }
  rewrite field_eqb_refl, node_eqb_refl in Hrow. cbn [andb] in Hrow.
  split.
  - destruct (s_any_source S a) eqn:E; [ | reflexivity ].
    apply (Hrow true). cbn in E. rewrite E. reflexivity.
  - intros Hs Hc. destruct (s_has_source S r) eqn:E; [ | reflexivity ].
    apply (Hrow true). apply orb_true_iff. right.
    apply existsb_exists. exists (a, s).
    split; [ apply (mem_pair_In _ _ _); exact Hs | ].
    cbn beta iota. rewrite node_eqb_refl. cbn [andb]. apply existsb_exists.
    exists (s, Field_return_of_call (Normal i), r).
    split; [ exact Hc | ]. cbn beta iota.
    rewrite node_eqb_refl, field_eqb_refl. exact E.
Qed.

Lemma witness_not_local (ep : closure_entry_point) :
  is_local (Field_call_witness ep) = false.
Proof.
  destruct (is_local (Field_call_witness ep)) eqn:E; [ | reflexivity ].
  destruct (Hloc _ E) as [(w & Hw) | (fs & Hfs)]; discriminate.
Qed.

(* A call of one group of arguments whose code is not known, read at the
   field of its arity, of whose result of rank [i] no source of the node
   read off its callee has a source: it never returns
   ([never_returns_of_call]). *)
Lemma one_group_never_returns (ep : closure_entry_point) (f a : node)
    (n nres i : nat) :
  (a, Field_call_witness ep, f) ∈ f_accessor F ->
  result_has_source_at F T (Field_call_witness ep) f i = false ->
  i < nres ->
  never_returns_of_call F S ep (get_direct_sources S f) [n] nres = true.
Proof.
  intros Ha Hno Hi.
  destruct (no_source_at _ f a a a i Hno Ha) as [Hasa _].
  pose proof (witness_not_local ep) as Hnl.
  cbn [never_returns_of_call]. unfold get_direct_sources.
  destruct (s_any_source S f) eqn:Ef.
  { rewrite (any_source_accessor _ _ _ closedU f _ a Ef Ha Hnl) in Hasa.
    discriminate. }
  unfold get_field_sources.
  match goal with
  | |- context [if existsb ?p (f_constructor F) then _ else _] =>
    destruct (existsb p (f_constructor F)) eqn:Eany
  end.
  { exfalso. apply existsb_exists in Eany as ([[x f'] y] & Hc & Ht).
    cbn beta iota in Ht.
    apply andb_true_iff in Ht as [Ht Hy].
    apply andb_true_iff in Ht as [Hx Hf'].
    apply andb_true_iff in Hx as [Hfx _].
    apply field_eqb_eq in Hf'. subst f'.
    pose proof (alias_accessor_source _ _ _ closedU f x _ a y Ha Hfx Ef Hc)
      as Hay.
    rewrite (any_source_alias _ _ _ closedU y a Hay Hy) in Hasa.
    discriminate. }
  cbn beta iota.
  apply existsb_exists. exists i.
  split; [ apply in_seq; lia | ].
  apply negb_true_iff. unfold result_has_source_of.
  apply Bool.not_true_iff_false. intros E.
  apply existsb_exists in E as ([[z f'] r] & Hc & Ht).
  cbn beta iota in Ht.
  apply andb_true_iff in Ht as [Ht Hr].
  apply andb_true_iff in Ht as [Hz Hf'].
  apply field_eqb_eq in Hf'. subst f'.
  apply existsb_exists in Hz as ([[x f'] y] & Hcx & Ht).
  cbn beta iota in Ht.
  apply andb_true_iff in Ht as [Ht _].
  apply andb_true_iff in Ht as [Ht Hyz].
  apply andb_true_iff in Ht as [Hx Hf'].
  apply andb_true_iff in Hx as [Hfx _].
  apply field_eqb_eq in Hf'. subst f'.
  pose proof (alias_accessor_source _ _ _ closedU f x _ a y Ha Hfx Ef Hcx)
    as Hay.
  assert (Hasy : s_any_source S y = false).
  { destruct (s_any_source S y) eqn:E; [ | reflexivity ].
    rewrite (any_source_alias _ _ _ closedU y a Hay E) in Hasa.
    discriminate. }
  pose proof (sources_alias _ _ _ closedU y a z Hyz Hasy Hay Hasa) as Haz.
  destruct (no_source_at _ f a z r i Hno Ha) as [_ Hnr].
  rewrite (Hnr Haz Hc) in Hr. discriminate.
Qed.

(* The call of the arm, direct or of known arity, never returns: its code,
   if known, has a result with no source (that of rank [i], which the
   call expects of it), and otherwise the witness that it reads has
   none. *)
Lemma known_never_returns (ap : apply_expr) (a : node) (i : nat) :
  ap_call_kind ap <> Indirect_unknown_arity ->
  (a, known, node_of_simple (ap_callee ap)) ∈ f_accessor F ->
  callee_result_has_source F T (node_of_simple (ap_callee ap)) i = false ->
  i < length (unarize (ap_return_arity ap)) ->
  (∀ (id : code_id) (cd : code_dep),
     code_id_actually_called F S ap = Some id -> (id, cd) ∈ cds ->
     length (unarize (ap_return_arity ap)) <= length (cd_return cd)) ->
  apply_never_returns F S cds (bound_code_metadata e0) ap = true.
Proof.
  intros Hk Hread Hno Hi Har.
  unfold callee_result_has_source in Hno.
  destruct (no_source_at _ _ a a a i Hno Hread) as [Hasa _].
  unfold apply_never_returns.
  destruct (code_id_actually_called F S ap) as [id | ] eqn:Ecall.
  - destruct (called_some F S ap id Ecall) as (ids & Eids & Esing).
    assert (Hid : id ∈ ids).
    { pose proof (proj1 (FlambdaMapProofs.keyed_get_singleton _ Pos.eqb
                           pos_eqb_decides id ids) Esing id) as H.
      rewrite Pos.eqb_refl in H.
      exact (proj1 (FlambdaMapProofs.keyed_among_in _ Pos.eqb pos_eqb_decides
                      id ids) H). }
    destruct (directly_inv F S _ ids id Eids Hid)
      as (a1 & s & Ha1 & Hs1 & Hhs & Hcid).
    destruct (kw_source F T cds Hknown a1 _ s Ha1 Hs1 Hhs)
      as (id' & cd & Hcd & ->).
    pose proof (proj2 (kw_constructor F T cds Hknown id' cd _ _ Hcd Hcid)
                  eq_refl) as Heq.
    injection Heq as <-.
    destruct (cds_code id cd Hcd) as (Hkw & (c & Hc) & _).
    pose proof (Har id cd eq_refl Hcd) as Hlen.
    destruct (nth_error (cd_return cd) i) as [r | ] eqn:Hr.
    2:{ apply nth_error_None in Hr. lia. }
    destruct Hkw as (_ & Hkw_res & _).
    pose proof (Hkw_res i r Hr) as Hcr.
    destruct (no_source_at _ _ a1 (cd_known_arity_call_witness cd) r i Hno
                Ha1)
      as [_ Hnr].
    pose proof (Hnr Hs1 Hcr) as Hnr'.
    destruct (lookup id (bound_code_metadata e0)) as [md | ] eqn:Emd.
    + unfold code_never_returns.
      rewrite (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd).
      apply existsb_exists. exists r.
      split; [ exact (nth_error_In _ _ Hr) | ].
      rewrite Hnr'. reflexivity.
    + exfalso. apply FlambdaMapProofs.lookup_none_not_in in Emd. apply Emd.
      rewrite (proj1 bound_code_metadata_codes e0), map_map.
      apply in_map_iff. exists (id, c). split; [ reflexivity | exact Hc ].
  - destruct (ap_callee ap) as [x | sym | c0] eqn:Hcallee;
      cbv zeta.
    3:{ cbn [node_of_simple] in Hread.
        pose proof (any_source_input _ _ _ closedU _ Hconsts) as Hac.
        rewrite (any_source_accessor _ _ _ closedU _ _ a Hac Hread
                   (known_not_local is_local Hloc))
          in Hasa.
        discriminate. }
    all: destruct (ap_call_kind ap) eqn:Hkind;
      try (exfalso; apply Hk; first [ reflexivity | exact Hkind ]);
      exact (one_group_never_returns Known_arity_code_pointer _ a _ _ i
               Hread Hno Hi).
Qed.

(* The call of the arm, of unknown arity with one group of arguments,
   never returns. *)
Lemma unknown_never_returns (ap : apply_expr) (a : node) (i : nat) :
  ap_call_kind ap = Indirect_unknown_arity ->
  length (ap_args_arity ap) = 1 ->
  (a, unknown, node_of_simple (ap_callee ap)) ∈ f_accessor F ->
  result_has_source_at F T unknown (node_of_simple (ap_callee ap)) i
  = false ->
  i < length (unarize (ap_return_arity ap)) ->
  apply_never_returns F S cds (bound_code_metadata e0) ap = true.
Proof.
  intros Hk Hlen Hread Hno Hi.
  destruct (no_source_at _ _ a a a i Hno Hread) as [Hasa _].
  assert (Hone : ∃ c, ap_args_arity ap = [c]).
  { destruct (ap_args_arity ap) as [ | c [ | c' cs]] eqn:E;
      cbn in Hlen; try discriminate.
    exists c. reflexivity. }
  destruct Hone as (c & Hc).
  unfold apply_never_returns, code_id_actually_called. rewrite Hk, Hc.
  cbn beta iota.
  destruct (ap_callee ap) as [x | sym | c0] eqn:Hcallee;
    cbv zeta.
  3:{ cbn [node_of_simple] in Hread.
      pose proof (any_source_input _ _ _ closedU _ Hconsts) as Hac.
      rewrite (any_source_accessor _ _ _ closedU _ _ a Hac Hread
                 (witness_not_local Unknown_arity_code_pointer))
        in Hasa.
      discriminate. }
  all: exact (one_group_never_returns Unknown_arity_code_pointer _ a _ _ i
                Hread Hno Hi).
Qed.

End Unit.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Lemma arm_unit (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (max_unbox_size : nat) (lambda_lifting : bool)
    (dominated : list (node * node)) (to_unbox : list node) :
  let F := traverse_unit flag u e in
  let code_deps := unit_code_deps flag u e in
  let unboxed := unit_unboxed F T dominated to_unbox e in
  is_local_slots_only is_local ->
  (∀ f : field, is_local f = false) ∨ projection_free e = true ->
  in_flow_fragment e = true ->
  least_solution is_local F T ->
  check_cc_closed F T cannot_change = true ->
  check_dominated T dominated = true ->
  check_to_unbox_fixed is_local F T cannot_change max_unbox_size
    lambda_lifting to_unbox
  = true ->
  FlambdaReaperCallingConventionsSpec.check_apply_widgets F T = true ->
  check_used_code_ids T = true ->
  check_argument_reads F = true ->
  check_known_witnesses F T code_deps = true ->
  conts_counted (unit_counts u) e = true ->
  plain_kinds e = true ->
  ∀ (ads : list decision) (ap : apply_expr),
    (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
    call_ok flag u e T cannot_change dominated to_unbox ads ap
    ∧ call_ok_indirect flag u e T cannot_change dominated to_unbox ads ap.
Proof.
  intros F code_deps unboxed Hloc Hslots Hfrag Hleast Hcc Hdom Hunbox Hwidgets
    Hcodes Hargs Hknown Hcount Hkinds ads ap Hin.
  destruct (unit_facts flag u e (∀ f : field, is_local f = false) Hfrag
              Hslots)
    as (noted & deps & Hedges & Hconsts & Hunit & Hnodup & Hids).
  assert (Hagree : agree flag u e T dominated to_unbox [] (unit_conts u)).
  { intros k ns Hk. left. split; [ reflexivity | ].
    intros n Hn. apply (any_usage_input _ _ _ (closedS _ _ _ Hleast)).
    apply (Hunit k n). unfold unit_cont_nodes. rewrite Hk. exact Hn. }
  exact (proj1 (calls_ok flag is_local u e T cannot_change max_unbox_size
                  lambda_lifting dominated to_unbox Hloc Hleast Hcc Hdom Hunbox
                  Hwidgets Hcodes Hargs Hknown _ noted deps Hconsts Hnodup Hids
                  (fun id c Hc =>
                     proj1 (unit_codes_edges _ _ _ _ _ _ _ _ _) e None
                       (unit_conts u) Hedges id c Hc))
           e None (unit_conts u) [] Hedges Hcount Hkinds
           ltac:(intros c Hc; discriminate) Hagree ads ap Hin).
Qed.

Theorem arm_needs_no_source : arm_needs_no_source_statement.
Proof.
  intros flag is_local u e T cannot_change max_unbox_size lambda_lifting
    dominated to_unbox F code_deps unboxed Hloc Hslots Hfrag Hleast Hcc Hdom
    Hunbox Hwidgets Hcodes Hargs Hknown Hcount Hkinds ads ap i fd kind Hin
    Hdirect Hfd Harm.
  exact (proj1 (proj2 (proj1 (arm_unit flag is_local u e T cannot_change
                                max_unbox_size lambda_lifting dominated
                                to_unbox Hloc Hslots Hfrag Hleast Hcc Hdom
                                Hunbox Hwidgets Hcodes Hargs Hknown Hcount
                                Hkinds ads ap Hin)
                         i fd kind Hdirect Hfd Harm))).
Qed.

Theorem arm_needs_no_source_auto : arm_needs_no_source_auto_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting
    dominated to_unbox F code_deps unboxed Hloc Hslots Hfrag Hleast Hcc Hdom
    Hunbox Hcodes Hargs Hknown Hcount Hkinds.
  apply (arm_needs_no_source Auto is_local u e T cannot_change max_unbox_size
           lambda_lifting dominated to_unbox Hloc Hslots Hfrag Hleast Hcc Hdom
           Hunbox); try assumption.
  apply (FlambdaReaperLeastProofs.least_apply_widgets is_local _ T Hloc
           Hleast).
  apply FlambdaReaperLeastProofs.traverse_widgets_read_only.
Qed.

Theorem kept_result_never_unboxed : kept_result_never_unboxed_statement.
Proof.
  intros flag is_local u e T cannot_change max_unbox_size lambda_lifting
    dominated to_unbox F code_deps unboxed Hloc Hslots Hfrag Hleast Hcc Hdom
    Hunbox Hwidgets Hcodes Hargs Hknown Hcount Hkinds ads ap i fields kind
    Hin Hdirect Hfd.
  destruct (arm_pair (apply_decision_at ads i kind) (Unbox fields)) eqn:Harm;
    [ exfalso | reflexivity ].
  exact (Bool.diff_true_false
           (proj1 (proj1 (arm_unit flag is_local u e T cannot_change
                            max_unbox_size lambda_lifting dominated to_unbox
                            Hloc Hslots Hfrag Hleast Hcc Hdom Hunbox Hwidgets
                            Hcodes Hargs Hknown Hcount Hkinds ads ap Hin)
                     i (Unbox fields) kind Hdirect Hfd Harm))).
Qed.

Theorem indirect_arm_needs_no_source : indirect_arm_needs_no_source_statement.
Proof.
  intros flag is_local u e T cannot_change max_unbox_size lambda_lifting
    dominated to_unbox F code_deps unboxed Hloc Hslots Hfrag Hleast Hcc Hdom
    Hunbox Hwidgets Hcodes Hargs Hknown Huw Hux Hcount Hkinds ads ap i fd
    kind Hin Hind Hlen Hfd Harm.
  exact (proj1 (proj2 (arm_unit flag is_local u e T cannot_change
                         max_unbox_size lambda_lifting dominated to_unbox Hloc
                         Hslots Hfrag Hleast Hcc Hdom Hunbox Hwidgets Hcodes
                         Hargs Hknown Hcount Hkinds ads ap Hin)
                  Huw Hux i fd kind Hind Hlen Hfd Harm)).
Qed.

(* The position of a decision of the callee is that of a result of the
   call. *)
Lemma callee_decision_rank (F : facts) (T : tables)
    (cannot_change : list node) (cds : finite_map code_dep)
    (unboxed : list (node * unboxed_fields)) (e : expr) (ap : apply_expr)
    (i : nat) (fd : decision) (kind : flambda_kind) :
  nth_error (unit_callee_decisions F T cannot_change cds unboxed e ap) i
  = Some (fd, kind) ->
  i < length (unarize (ap_return_arity ap)).
Proof.
  unfold unit_callee_decisions, callee_decisions. intros Hfd.
  apply nth_error_Some.
  destruct (apply_change_u _ _ _ _ _ _ _ ap) as [ | pds rds].
  - rewrite nth_error_map in Hfd.
    destruct (nth_error (unarize (ap_return_arity ap)) i); [ | discriminate ].
    discriminate.
  - apply nth_error_combine in Hfd as [_ H]. rewrite H. discriminate.
Qed.

Lemma decision_at_unbox (ads : list decision) (i : nat)
    (fields : unboxed_fields) (kind : flambda_kind) :
  nth_error ads i = Some (Unbox fields) ->
  apply_decision_at ads i kind = Unbox fields.
Proof. intros H. unfold apply_decision_at. rewrite H. reflexivity. Qed.

Theorem fix_arm_never_returns : fix_arm_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change max_unbox_size lambda_lifting
    dominated to_unbox F code_deps unboxed Hloc Hslots Hfrag Hleast Hcc Hdom
    Hunbox Hwidgets Hcodes Hargs Hknown Hcount Hkinds ads ap i fields fd
    kind Hin Hdirect Har Hfd Hads Harm.
  rewrite <- (decision_at_unbox ads i fields kind Hads) in Harm.
  destruct (proj1 (arm_unit flag is_local u e T cannot_change max_unbox_size
                     lambda_lifting dominated to_unbox Hloc Hslots Hfrag Hleast
                     Hcc Hdom Hunbox Hwidgets Hcodes Hargs Hknown Hcount Hkinds
                     ads ap Hin)
              i fd kind Hdirect Hfd Harm)
    as (_ & Hno & a & Hread).
  destruct (unit_facts flag u e (∀ f : field, is_local f = false) Hfrag
              Hslots)
    as (noted & deps & Hedges & Hconsts & _ & Hnodup & Hids).
  exact (known_never_returns flag is_local u e T Hloc Hleast Hknown _ noted
           deps Hconsts Hnodup Hids
           (fun id c Hc =>
              proj1 (unit_codes_edges _ _ _ _ _ _ _ _ _) e None
                (unit_conts u) Hedges id c Hc)
           ap a i Hdirect Hread Hno
           (callee_decision_rank _ _ _ _ _ _ ap i fd kind Hfd) Har).
Qed.

Theorem indirect_fix_arm_never_returns :
  indirect_fix_arm_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change max_unbox_size lambda_lifting
    dominated to_unbox F code_deps unboxed Hloc Hslots Hfrag Hleast Hcc Hdom
    Hunbox Hwidgets Hcodes Hargs Hknown Huw Hux Hcount Hkinds ads ap i fields
    fd kind Hin Hind Hlen Hfd Hads Harm.
  rewrite <- (decision_at_unbox ads i fields kind Hads) in Harm.
  destruct (proj2 (arm_unit flag is_local u e T cannot_change max_unbox_size
                     lambda_lifting dominated to_unbox Hloc Hslots Hfrag Hleast
                     Hcc Hdom Hunbox Hwidgets Hcodes Hargs Hknown Hcount Hkinds
                     ads ap Hin)
              Huw Hux i fd kind Hind ltac:(lia) Hfd Harm)
    as (Hno & a & Hread).
  destruct (unit_facts flag u e (∀ f : field, is_local f = false) Hfrag
              Hslots)
    as (noted & deps & Hedges & Hconsts & _ & Hnodup & Hids).
  exact (unknown_never_returns flag is_local u e T Hloc Hleast Hconsts ap a
           i Hind Hlen Hread Hno
           (callee_decision_rank _ _ _ _ _ _ ap i fd kind Hfd)).
Qed.
