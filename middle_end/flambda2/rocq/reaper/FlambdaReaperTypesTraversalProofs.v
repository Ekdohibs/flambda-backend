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

(* The proofs of FlambdaReaperTypesTraversalSpec.v. Nothing here needs
   review.

   [reprs_of_sets_derived]: the fold of [changed_representations] keeps
   [rinv]: a closure that has a representation has that of the set of a
   closure [x] of [tcr], at its own function slot, and so do the other
   closures of that set, which are seen. A closure of the set of [x] has
   no representation before [x] is met: the closure that gave it one
   would hold [x], which would then be seen.

   The traversal: the slot facts of the graph are those of the groups of
   FlambdaReaperClosuresExactProofs.v ([traverse_unit_groups]), which are
   functional and whose allocations hold the same slots. The traversal
   keeps [minv]: a function slot of a [Let] of a closure is held by each
   of the allocations of its set, which are distinct and include the
   variable of the [Let]; one of a static closure is a symbol of its
   set.

   [block_fields_of_one_kind_traversal]: the steps of the traversal add
   no constructor at a field of a block of a kind other than value
   ([Kinds.ko_ok], the [ns_ok] of FlambdaReaperClosuresExactProofs.v for
   [Kinds.bad_fact]).

   [changed_closures_one_allocation_unaliased]: by induction on
   [derivable], each tuple of a closure is as [ok_atom] says. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperSolutionProofs FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperTypesSpec FlambdaReaperTypesTraversalSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxing FlambdaReaperUnboxingSpec
  FlambdaReaperUnboxingProofs FlambdaReaperTypesProofs
  FlambdaReaperClosuresExactProofs FlambdaReaperFlowProofs.
Import ListNotations.

Module U := FlambdaReaperUnboxing.
Module US := FlambdaReaperUnboxingSpec.
Module TP := FlambdaReaperTypesProofs.
Module CE := FlambdaReaperClosuresExactProofs.

(* ======================================================================== *)
(* The representations of the unit                                          *)
(* ======================================================================== *)

Lemma rename_keys (L : list function_slot) :
  ∀ n : positive, map fst (fst (U.rename_slots L n)) = L.
Proof.
  induction L as [ | fs L IH]; intros n; [ reflexivity | ].
  cbn [U.rename_slots].
  destruct (U.rename_slots L (Pos.succ n)) as [rest n'] eqn:E.
  cbn [fst map]. f_equal. rewrite <- (IH (Pos.succ n)), E. reflexivity.
Qed.

Lemma rename_nodup (L : list function_slot) :
  ∀ n : positive,
    NoDup (map snd (fst (U.rename_slots L n)))
    ∧ ∀ p : positive,
        p ∈ map snd (fst (U.rename_slots L n)) -> (n <= p)%positive.
Proof.
  induction L as [ | fs L IH]; intros n.
  - split; [ constructor | intros p [] ].
  - cbn [U.rename_slots].
    destruct (IH (Pos.succ n)) as [Hnd Hge].
    destruct (U.rename_slots L (Pos.succ n)) as [rest n'] eqn:E.
    cbn [fst snd map] in *. split.
    + constructor; [ | exact Hnd ].
      intros Hin. pose proof (Hge n Hin). lia.
    + intros p [<- | Hin]; [ lia | ]. pose proof (Hge p Hin). lia.
Qed.

Lemma keyed_lookup_some (k : node) {A : Type} (v : A)
    (l : list (node * A)) :
  (k, v) ∈ l -> ∃ v' : A, Keyed.lookup node_eqb k l = Some v'.
Proof.
  induction l as [ | [k' w] l IH]; intros Hin; [ destruct Hin | ].
  cbn [Keyed.lookup].
  destruct (node_eqb k k') eqn:E; [ eexists; reflexivity | ].
  destruct Hin as [[= <- <-] | Hin]; [ | exact (IH Hin) ].
  rewrite node_eqb_refl in E. discriminate E.
Qed.

Lemma mem_node_app_l (n : node) (l l' : list node) :
  mem_node n l = true -> mem_node n (l ++ l') = true.
Proof.
  intros H. apply mem_node_In. apply in_or_app. left.
  apply mem_node_In. exact H.
Qed.

Section Reprs.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Variable cm : finite_map variable.
Variable dominated : list (node * node).
Variable to_unbox : list node.
Variable tcr : list node.

Abbreviation G := (graph_of_facts F).
Abbreviation sol := (solution_of_tables T).

Hypothesis Hc : closed is_local G sol.
Hypothesis Hsets : function_slots_of_sets G.
Hypothesis Hmut : function_slots_mutual G.
Hypothesis Hdom : U.check_dominated T dominated = true.
Hypothesis Hone : changed_closures_one_allocation G sol tcr.

Abbreviation fsf a g b := (g_constructor G a (Field_function_slot g) b).

(* What dominates a closure is the closure. *)
Lemma dominated_self (f c b : node) (g : function_slot) :
  fsf c g b -> (f, c) ∈ dominated -> f = c.
Proof.
  intros Hcb Hd.
  destruct (proj1 (FlambdaReaperUnboxingProofs.check_dominated_sound T
                     dominated Hdom f c)
              (proj2 (mem_pair_In f c dominated) Hd))
    as (Hsrc & Hhs & Hnm).
  assert (Hna : s_any_source sol c = false).
  { apply not_true_iff_false. intros H. apply Hnm. left. exact H. }
  pose proof (sources_constructor _ _ _ Hc b (Field_function_slot g) c Hcb
                Hna) as Hcc.
  pose proof (has_source_function_slot _ _ _ Hc b (Field_function_slot g) c
                Hcb eq_refl) as Hhc.
  destruct (node_eq_dec f c) as [E | Hne]; [ exact E | ].
  exfalso. apply Hnm. right. exists f, c. auto 6.
Qed.

(* A closure of the set of a closure of [tcr] dominates itself. *)
Lemma self_dominated (x c : node) (g : function_slot) :
  x ∈ tcr -> fsf x g c -> (c, c) ∈ dominated.
Proof.
  intros Hx Hxc. pose proof (Hone x c g Hx Hxc) as Hnm.
  destruct (Hmut x c g Hxc) as (g' & Hcx).
  assert (Hna : s_any_source sol c = false).
  { apply not_true_iff_false. intros H. apply Hnm. left. exact H. }
  pose proof (sources_constructor _ _ _ Hc x (Field_function_slot g') c Hcx
                Hna) as Hcc.
  pose proof (has_source_function_slot _ _ _ Hc x (Field_function_slot g') c
                Hcx eq_refl) as Hhc.
  apply mem_pair_In.
  apply (proj2 (FlambdaReaperUnboxingProofs.check_dominated_sound T dominated
                  Hdom c c)).
  split; [ exact Hcc | split; [ exact Hhc | exact Hnm ] ].
Qed.

Lemma sfs_iff (x b : node) (g : function_slot) :
  (g, b) ∈ U.set_function_slots F x <-> fsf x g b.
Proof.
  split; [ apply TP.set_function_slots_in | ].
  intros H. destruct (TP.set_function_slots_has F x g b H) as (b' & Hb').
  rewrite ((proj1 Hsets) x b b' g H (TP.set_function_slots_in F x g b' Hb')).
  exact Hb'.
Qed.

(* The invariant of the fold. *)
Definition rinv (acc : list (node * U.closure_repr)) (seen : list node)
    : Prop :=
  (∀ (c : node) (r : U.closure_repr) (g : function_slot) (b : node),
     fsf c g b -> Keyed.lookup node_eqb c acc = Some r ->
     ∃ x : node,
       x ∈ tcr
       ∧ (U.cr_current r, c) ∈ U.set_function_slots F x
       ∧ map fst (U.cr_function_slots r)
         = map fst (U.set_function_slots F x)
       ∧ ∀ (h : function_slot) (d : node),
           (h, d) ∈ U.set_function_slots F x ->
           Keyed.lookup node_eqb d acc
           = Some (U.Closure_repr (U.cr_fields r) (U.cr_function_slots r) h)
           ∧ mem_node d seen = true)
  ∧ ∀ p : node * U.closure_repr,
      p ∈ acc -> NoDup (map snd (U.cr_function_slots (snd p))).

Lemma in_new (sx : list (function_slot * node)) (fields : U.unboxed_fields)
    (fss : list (function_slot * function_slot)) (c : node)
    (r : U.closure_repr) :
  (c, r)
  ∈ flat_map
      (fun '(fs, f) =>
         flat_map
           (fun '(a, c) =>
              if node_eqb a f then [(c, U.Closure_repr fields fss fs)] else [])
           dominated)
      sx
  <-> ∃ (fs : function_slot) (f : node),
        (fs, f) ∈ sx ∧ (f, c) ∈ dominated ∧ r = U.Closure_repr fields fss fs.
Proof.
  rewrite in_flat_map. split.
  - intros ([fs f] & Hfs & Hin). apply in_flat_map in Hin as ([a c'] & Hd & H).
    destruct (node_eqb a f) eqn:E; [ | destruct H ].
    apply node_eqb_eq in E. subst a.
    destruct H as [[= <- <-] | []]. exists fs, f. auto.
  - intros (fs & f & Hfs & Hd & ->). exists (fs, f).
    split; [ exact Hfs | ]. apply in_flat_map. exists (f, c).
    split; [ exact Hd | ]. rewrite node_eqb_refl. left. reflexivity.
Qed.

(* A closure [x] of [tcr] not seen gives its set a representation. *)
Lemma rinv_step (acc : list (node * U.closure_repr)) (seen : list node)
    (x : node) (sx : list (function_slot * node)) (fields : U.unboxed_fields)
    (n1 n2 : positive) (fss : list (function_slot * function_slot)) :
  rinv acc seen -> x ∈ tcr -> mem_node x seen = false ->
  U.set_function_slots F x = sx ->
  U.rename_slots (map fst sx) n1 = (fss, n2) ->
  rinv
    (acc
     ++ flat_map
          (fun '(fs, f) =>
             flat_map
               (fun '(a, c) =>
                  if node_eqb a f
                  then [(c, U.Closure_repr fields fss fs)]
                  else [])
               dominated)
          sx)
    (seen ++ map snd sx).
Proof.
  intros [Hinv1 Hinv2] Hx Hseen Hsx Hren. subst sx.
  set (sx := U.set_function_slots F x) in *.
  set (new := flat_map _ sx).
  assert (Hfss : map fst fss = map fst sx).
  { pose proof (rename_keys (map fst sx) n1) as E. rewrite Hren in E.
    exact E. }
  assert (Hnd : NoDup (map snd fss)).
  { pose proof (proj1 (rename_nodup (map fst sx) n1)) as E.
    rewrite Hren in E. exact E. }
  split.
  - intros c r g b Hcb Hlk. rewrite keyed_lookup_app in Hlk.
    destruct (Keyed.lookup node_eqb c acc) as [r0 | ] eqn:Ho.
    + injection Hlk as <-.
      destruct (Hinv1 c r0 g b Hcb Ho) as (x0 & Hx0 & Hcur & Hmap & Hall).
      exists x0. split; [ exact Hx0 | split; [ exact Hcur | ] ].
      split; [ exact Hmap | ]. intros h d Hhd.
      destruct (Hall h d Hhd) as [Hd1 Hd2].
      rewrite keyed_lookup_app, Hd1.
      split; [ reflexivity | apply mem_node_app_l; exact Hd2 ].
    + apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hlk.
      apply in_new in Hlk as (fs & f & Hfs & Hd & ->).
      rewrite (dominated_self f c b g Hcb Hd) in Hfs.
      exists x. cbn [U.cr_current U.cr_fields U.cr_function_slots].
      split; [ exact Hx | split; [ exact Hfs | split; [ exact Hfss | ] ] ].
      intros h d Hhd. pose proof (proj1 (sfs_iff x d h) Hhd) as Hxd.
      destruct (Hmut x d h Hxd) as (h' & Hdx).
      split.
      * rewrite keyed_lookup_app.
        destruct (Keyed.lookup node_eqb d acc) as [r0 | ] eqn:Hd0.
        -- exfalso.
           destruct (Hinv1 d r0 h' x Hdx Hd0)
             as (x0 & _ & Hcur0 & _ & Hall0).
           pose proof (proj1 (sfs_iff x0 d _) Hcur0) as Hx0d.
           pose proof (proj2 ((proj2 (proj2 Hsets)) x0 d x _ h' Hx0d) Hdx)
             as Hx0x.
           destruct (Hall0 h' x (proj2 (sfs_iff x0 x h') Hx0x)) as [_ Hm].
           rewrite Hm in Hseen. discriminate Hseen.
        -- assert (Hdd : (d, d) ∈ dominated)
             by exact (self_dominated x d h Hx Hxd).
           assert (Hinn : (d, U.Closure_repr fields fss h) ∈ new)
             by (apply in_new; exists h, d; auto).
           destruct (keyed_lookup_some d _ new Hinn) as (r1 & Hr1).
           rewrite Hr1. f_equal.
           apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hr1.
           apply in_new in Hr1 as (fs1 & f1 & Hfs1 & Hd1 & ->).
           rewrite (dominated_self f1 d x h' Hdx Hd1) in Hfs1.
           f_equal.
           exact ((proj1 (proj2 Hsets)) x d fs1 h
                    (proj1 (sfs_iff x d fs1) Hfs1) Hxd).
      * apply mem_node_In. apply in_or_app. right.
        apply in_map_iff. exists (h, d). auto.
  - intros p Hp. apply in_app_or in Hp as [Hp | Hp]; [ exact (Hinv2 p Hp) | ].
    destruct p as [c r]. apply in_new in Hp as (fs & f & _ & _ & ->).
    exact Hnd.
Qed.

Lemma rinv_fold :
  ∀ next : positive,
    ∃ seen : list node,
      rinv (fst (U.changed_representations F T cm dominated to_unbox
                   is_local tcr next))
        seen.
Proof.
  intros next. unfold U.changed_representations.
  match goal with
  | |- context [List.fold_left ?h tcr _] => set (step := h)
  end.
  assert (Hgen :
    ∀ (l : list node) (acc : list (node * U.closure_repr)) (seen : list node)
      (n : positive),
      (∀ y : node, y ∈ l -> y ∈ tcr) -> rinv acc seen ->
      rinv (fst (fst (List.fold_left step l (acc, seen, n))))
        (snd (fst (List.fold_left step l (acc, seen, n))))).
  { induction l as [ | x l IH]; intros acc seen n Hl Hinv; [ exact Hinv | ].
    cbn [List.fold_left].
    destruct (step (acc, seen, n) x) as [[acc' seen'] n'] eqn:Hstep.
    apply IH; [ intros y Hy; apply Hl; right; exact Hy | ].
    assert (Hx : x ∈ tcr) by (apply Hl; left; reflexivity).
    unfold step in Hstep. cbv beta match in Hstep.
    destruct (mem_node x seen) eqn:Hseen;
      [ injection Hstep as <- <- _; exact Hinv | ].
    destruct (U.set_function_slots F x) as [ | [fs0 f00] l0] eqn:Hsx;
      [ injection Hstep as <- <- _; exact Hinv | ].
    destruct (U.mk_unboxed_fields F T cm dominated to_unbox
                (S (length to_unbox)) x
                (U.get_fields_of_constructors F T is_local
                   (map snd ((fs0, f00) :: l0)))
                n)
      as [fields n1] eqn:Hmk.
    destruct (U.rename_slots (map fst ((fs0, f00) :: l0)) n1)
      as [fss n2] eqn:Hren.
    injection Hstep as <- <- _.
    exact (rinv_step acc seen x _ fields n1 n2 fss Hinv Hx Hseen Hsx Hren). }
  destruct (List.fold_left step tcr ([], [], next)) as [[acc seen] n] eqn:Hf.
  cbn [fst].
  assert (H0 : rinv [] []).
  { split; [ intros c r g b _ H; discriminate H | intros p [] ]. }
  pose proof (Hgen tcr [] [] next (fun y H => H) H0) as H.
  rewrite Hf in H. exists seen. exact H.
Qed.

End Reprs.

Theorem reprs_of_sets_derived : reprs_of_sets_derived_statement.
Proof.
  intros is_local F T dominated to_unbox tcr e Hleast Hsets Hmut Hdom Hone.
  assert (Hc : closed is_local (graph_of_facts F) (solution_of_tables T))
    by exact (TP.least_closed_tables is_local F T (or_introl Hleast)).
  unfold U.unit_changed_representations.
  match goal with
  | |- context [U.changed_representations _ _ ?cm _ _ _ _ ?n] =>
    destruct (rinv_fold is_local F T cm dominated to_unbox tcr Hc Hsets Hmut
                Hdom Hone n)
      as (seen & Hinv1 & Hinv2)
  end.
  set (reprs := fst _) in *.
  split.
  - intros a b g Hab.
    destruct (Keyed.lookup node_eqb a reprs) as [r | ] eqn:Ha.
    + destruct (Hinv1 a r g b Hab Ha) as (x & _ & Hcur & Hmap & Hall).
      pose proof (proj1 (sfs_iff F Hsets x a _) Hcur) as Hxa.
      pose proof (proj2 ((proj2 (proj2 Hsets)) x a b _ g Hxa) Hab) as Hxb.
      pose proof (proj2 (sfs_iff F Hsets x b g) Hxb) as Hgb.
      destruct (Hall g b Hgb) as [Hb _].
      rewrite Hb. cbn [U.cr_fields U.cr_function_slots U.cr_current].
      split; [ reflexivity | split; [ reflexivity | ] ].
      split; [ reflexivity | ].
      assert (Hg : g ∈ map fst (U.cr_function_slots r)).
      { rewrite Hmap. apply in_map_iff. exists (g, b).
        split; [ reflexivity | exact Hgb ]. }
      rewrite is_bound_among. apply among_In. exact Hg.
    + destruct (Keyed.lookup node_eqb b reprs) as [r' | ] eqn:Hb;
        [ | exact I ].
      destruct (Hmut a b g Hab) as (g0 & Hba).
      destruct (Hinv1 b r' g0 a Hba Hb) as (x & _ & Hcur & _ & Hall).
      pose proof (proj1 (sfs_iff F Hsets x b _) Hcur) as Hxb.
      pose proof (proj2 ((proj2 (proj2 Hsets)) x b a _ g0 Hxb) Hba) as Hxa.
      destruct (Hall g0 a (proj2 (sfs_iff F Hsets x a g0) Hxa)) as [Ha' _].
      rewrite Ha in Ha'. discriminate Ha'.
  - intros a r Hlk.
    apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hlk.
    exact (Hinv2 (a, r) Hlk).
Qed.

(* ======================================================================== *)
(* The static sets                                                          *)
(* ======================================================================== *)

(* Each binding of a symbol of a term whose sets are well formed is whole
   in its group. *)
Lemma sym_binding_whole :
  (∀ e : expr,
     sets_well_formed e = true ->
     ∀ (t : symbol) (c : static_const),
       (t, c) ∈ CE.sym_bindings e ->
       ∃ bs : list static_binding, static_set_whole bs (Bind_symbol t c) = true)
  ∧ ∀ h : continuation_handler,
      match h with
      | Handler _ _ handler =>
        sets_well_formed handler = true ->
        ∀ (t : symbol) (c : static_const),
          (t, c) ∈ CE.sym_bindings handler ->
          ∃ bs : list static_binding,
            static_set_whole bs (Bind_symbol t c) = true
      end.
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [sets_well_formed CE.sym_bindings].
  - intros x k d body IH Hwf. apply andb_true_iff in Hwf as [_ Hwf].
    exact (IH Hwf).
  - intros bindings body Hcodes IH Hwf t c Hin.
    apply andb_true_iff in Hwf as [Hwf Hbody].
    apply andb_true_iff in Hwf as [Hwf Hcodes_wf].
    apply andb_true_iff in Hwf as [Hwhole _].
    rewrite forallb_forall in Hwhole, Hcodes_wf.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH Hbody t c Hin) ].
    apply in_flat_map in Hin as (b & Hb & Hin).
    destruct b as [id cd | id | t' c']; cbn in Hin.
    + destruct cd as [params mc ret exn md cbody]. cbn in Hin.
      pose proof (Hcodes_wf _ Hb) as Hc. cbn in Hc.
      exact (Hcodes id _ Hb Hc t c Hin).
    + destruct Hin.
    + destruct Hin as [[= <- <-] | []]. exists bindings.
      exact (Hwhole _ Hb).
  - intros k h body IHh IH Hwf t c Hin. destruct h as [params md handler].
    cbn [handler_sets_well_formed CE.handler_sym_bindings] in Hwf, Hin.
    apply andb_true_iff in Hwf as [Hh Hb].
    apply in_app_or in Hin as [Hin | Hin];
      [ exact (IHh Hh t c Hin) | exact (IH Hb t c Hin) ].
  - intros handlers body IHh IH Hwf t c Hin.
    apply andb_true_iff in Hwf as [Hhs Hb]. rewrite forallb_forall in Hhs.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH Hb t c Hin) ].
    apply in_flat_map in Hin as ([k h] & Hkh & Hin).
    destruct h as [params md handler]. cbn in Hin.
    pose proof (Hhs _ Hkh) as Hh. cbn in Hh.
    exact (IHh k _ Hkh Hh t c Hin).
  - intros ap _ t c [].
  - intros ac _ t c [].
  - intros scrutinee arms _ t c [].
  - intros _ t c [].
  - intros params metadata handler IH. exact IH.
Qed.

(* The symbol of a static closure is the one its set has at its slot. *)
Lemma static_lookup (e : expr) (t : symbol) (slot : function_slot)
    (cs : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) :
  sets_well_formed e = true ->
  (t, Static_const_closure_of_set slot cs decls vs) ∈ CE.sym_bindings e ->
  lookup slot cs = Some t.
Proof.
  intros Hwf Hin.
  destruct (proj1 sym_binding_whole e Hwf _ _ Hin) as (bs & Hw).
  cbn [static_set_whole] in Hw.
  apply andb_true_iff in Hw as [Hw _]. apply andb_true_iff in Hw as [_ Hw].
  destruct (lookup slot cs) as [s' | ]; [ | discriminate Hw ].
  apply Pos.eqb_eq in Hw. subst s'. reflexivity.
Qed.

(* ======================================================================== *)
(* The closures of one allocation                                           *)
(* ======================================================================== *)

Section OneAllocation.

Variable is_local : field -> bool.
Variable G : graph.
Variable S : solution.

Hypothesis Hfun :
  ∀ (a b b' : node) (g : function_slot),
    g_constructor G a (Field_function_slot g) b ->
    g_constructor G a (Field_function_slot g) b' -> b = b'.
Hypothesis Hmut : function_slots_mutual G.
Hypothesis Hu : closures_unaliased G.

Definition is_closure (b : node) : Prop :=
  ∃ (x : node) (g : function_slot), g_constructor G x (Field_function_slot g) b.

(* A read into a closure is at its slot, off a closure that holds it. *)
Lemma read_closure (t base : node) (f : field) :
  is_closure t -> g_accessor G t f base ->
  ∃ h : function_slot,
    f = Field_function_slot h
    ∧ g_constructor G base (Field_function_slot h) t ∧ is_closure base.
Proof.
  intros (x & g & Hx) Ha.
  destruct (Hu x t g Hx) as (_ & _ & _ & _ & _ & _ & _ & _ & Hacc).
  destruct (Hacc f base Ha) as (h & -> & Hb).
  destruct (Hmut base t h Hb) as (g' & Hb').
  exists h. split; [ reflexivity | split; [ exact Hb | exists t, g' ] ].
  exact Hb'.
Qed.

(* What the rules derive of the closures: an alias into one is from
   itself, it has no [any_source] and no [reading_field], its sources
   are itself, and it is a use of itself only. *)
Definition ok_atom (a : atom) : Prop :=
  match a with
  | Atom_alias t f => is_closure t -> f = t
  | Atom_any_source n => ¬ is_closure n
  | Atom_sources n y => is_closure n -> y = n
  | Atom_usages n u => is_closure u -> n = u
  | Atom_reading_field _ n => ¬ is_closure n
  | _ => True
  end.

Ltac contra_into :=
  let Hc := fresh "Hc" in
  intros Hc; exfalso; destruct Hc as (x0 & g0 & Hx0);
  destruct (Hu x0 _ g0 Hx0) as (U1 & U2 & U3 & U4 & U5 & U6 & U7 & U8 & _);
  first [ eapply U1; eassumption | eapply U2; eassumption
        | eapply U3; eassumption | eapply U4; eassumption
        | apply U5; assumption | apply U6; assumption
        | eapply U7; eassumption | eapply U8; eassumption ].

Ltac alias_read :=
  let Hc := fresh "Hc" in
  intros Hc;
  match goal with
  | Ha : g_accessor G ?t ?rel ?X, IH : is_closure ?X -> ?Y = ?X,
    Hk : g_constructor G ?Y ?rel ?f |- ?f = ?t =>
    let h := fresh "h" in let Hb := fresh "Hb" in let HX := fresh "HX" in
    destruct (read_closure t X rel Hc Ha) as (h & -> & Hb & HX);
    rewrite (IH HX) in Hk; exact (Hfun _ _ _ _ Hk Hb)
  end.

Ltac any_alias :=
  let Hc := fresh "Hc" in
  intros Hc;
  match goal with
  | IH : is_closure ?t -> ?f = ?t, IH2 : ¬ is_closure ?f |- False =>
    apply IH2; rewrite (IH Hc); exact Hc
  end.

Ltac any_read :=
  let Hc := fresh "Hc" in
  intros Hc;
  match goal with
  | Ha : g_accessor G ?t ?rel ?X, IH : ¬ is_closure ?X |- False =>
    let HX := fresh "HX" in
    destruct (read_closure t X rel Hc Ha) as (_ & _ & _ & HX);
    exact (IH HX)
  end.

Ltac usages_alias :=
  let Hc := fresh "Hc" in
  intros Hc;
  match goal with
  | IH1 : is_closure ?u -> ?t = ?u, IH2 : is_closure ?t -> ?f = ?t
    |- ?f = ?u =>
    let E := fresh "E" in
    pose proof (IH1 Hc) as E; rewrite <- E in Hc |- *; exact (IH2 Hc)
  end.

Ltac sources_alias :=
  let Hc := fresh "Hc" in
  intros Hc;
  match goal with
  | IH2 : is_closure ?t -> ?f = ?t, IH1 : is_closure ?f -> ?s = ?f
    |- ?s = ?t =>
    let E := fresh "E" in
    pose proof (IH2 Hc) as E; rewrite E in IH1; exact (IH1 Hc)
  end.

Lemma ok_derivable (a : atom) : derivable is_local G S a -> ok_atom a.
Proof.
  intros Hd. induction Hd; cbn [ok_atom] in *;
    solve [ exact I | intros; reflexivity | contra_into | alias_read
          | intros ?; contradiction | any_alias | any_read | usages_alias
          | sources_alias ].
Qed.

(* A closure has no [multiple_allocation_points] in a solution whose
   tuples the rules derive. *)
Lemma closure_one_allocation (x b : node) (g : function_slot) :
  (∀ a : atom, holds S a = true -> derivable is_local G S a) ->
  g_constructor G x (Field_function_slot g) b ->
  ¬ US.multiple_allocation_points_of S b.
Proof.
  intros Hder Hxb.
  assert (Hc : is_closure b) by (exists x, g; exact Hxb).
  intros [Hany | (y & z & Hy & _ & Hz & _ & Hne)].
  - exact (ok_derivable _ (Hder (Atom_any_source b) Hany) Hc).
  - apply Hne.
    rewrite (ok_derivable _ (Hder (Atom_sources b y) Hy) Hc),
      (ok_derivable _ (Hder (Atom_sources b z) Hz) Hc).
    reflexivity.
Qed.

End OneAllocation.

Theorem changed_closures_one_allocation_unaliased :
  changed_closures_one_allocation_unaliased_statement.
Proof.
  intros is_local F T tcr [_ Hder] Hsets Hmut Hu x b g _ Hxb.
  exact (closure_one_allocation is_local _ _ (proj1 Hsets) Hmut Hu x b g Hder
           Hxb).
Qed.

Lemma forallb_in {A : Type} (p : A -> bool) (l : list A) (x : A) :
  forallb p l = true -> x ∈ l -> p x = true.
Proof. intros H Hx. rewrite forallb_forall in H. exact (H x Hx). Qed.

Theorem check_closures_unaliased_sound :
  check_closures_unaliased_sound_statement.
Proof.
  intros F Hck x b g Hxb.
  pose proof (forallb_in _ _ _ Hck Hxb) as H. cbn beta iota in H.
  cbn [is_function_slot negb orb] in H.
  repeat (apply andb_true_iff in H as [H ?]).
  rename H into Ha.
  match goal with
  | H1 : forallb _ (f_propagate F) = true,
    H2 : forallb _ (f_alias_if_any_source F) = true,
    H3 : forallb _ (f_parameter F) = true,
    H4 : negb (mem_node b (f_any_source F)) = true,
    H5 : negb (mem_node b (f_zero_alloc_source F)) = true,
    H6 : forallb _ (f_any_source_if_used F) = true,
    H7 : forallb _ (f_use F) = true,
    H8 : forallb _ (f_accessor F) = true |- _ =>
    rename H1 into Hp, H2 into Hi, H3 into Hpa, H4 into Has, H5 into Hz,
      H6 into Hiu, H7 into Hu, H8 into Hacc
  end.
  assert (Hne : ∀ t : node, negb (node_eqb t b) = true -> t <> b).
  { intros t E ->. rewrite node_eqb_refl in E. discriminate E. }
  assert (Hnm : ∀ l : list node, negb (mem_node b l) = true -> b ∉ l).
  { intros l E Hin. apply mem_node_In in Hin. rewrite Hin in E.
    discriminate E. }
  cbn [graph_of_facts g_alias g_propagate g_alias_if_any_source g_parameter
       g_any_source g_zero_alloc_source g_any_source_if_used g_use
       g_accessor g_constructor].
  split; [ intros from Hin; exact (Hne _ (forallb_in _ _ _ Ha Hin) eq_refl)
         | ].
  split; [ intros c from Hin; exact (Hne _ (forallb_in _ _ _ Hp Hin) eq_refl)
         | ].
  split; [ intros c from Hin; exact (Hne _ (forallb_in _ _ _ Hi Hin) eq_refl)
         | ].
  split;
    [ intros base i Hin; exact (Hne _ (forallb_in _ _ _ Hpa Hin) eq_refl)
    | ].
  split; [ exact (Hnm _ Has) | split; [ exact (Hnm _ Hz) | ] ].
  split; [ intros c Hin; exact (Hne _ (forallb_in _ _ _ Hiu Hin) eq_refl) | ].
  split; [ intros from Hin; exact (Hne _ (forallb_in _ _ _ Hu Hin) eq_refl)
         | ].
  intros f base Hin. pose proof (forallb_in _ _ _ Hacc Hin) as E.
  cbn beta iota in E. rewrite node_eqb_refl in E. cbn [negb orb] in E.
  apply andb_true_iff in E as [Ef E].
  apply existsb_exists in E as ([[base' f'] from] & Hc & E).
  apply andb_true_iff in E as [E Eb]. apply andb_true_iff in E as [Eb' Ef'].
  apply node_eqb_eq in Eb, Eb'. apply field_eqb_eq in Ef'. subst.
  destruct f; try discriminate Ef. exists f. split; [ reflexivity | exact Hc ].
Qed.

Theorem closures_unaliased_examples : closures_unaliased_examples_statement.
Proof. vm_compute. auto. Qed.

(* ======================================================================== *)
(* The function slots of the traversal                                      *)
(* ======================================================================== *)

(* The closures of a [Closure_of_set]: [x] at [slot], the variable of its
   projection at a slot projected, a node made at each other slot. *)
Lemma closures_of_set_exact (x : variable) (slot : function_slot)
    (others : list (function_slot * variable)) :
  ∀ (decls : list (function_slot * code_id)) (a : acc)
    (cl : list (function_slot * node * code_id)) (a' : acc),
    closures_of_set x slot others decls a = (cl, a') ->
    map (fun '(f, _, _) => f) cl = map fst decls
    ∧ (acc_next a <= acc_next a')%positive
    ∧ (∀ (f : function_slot) (N : node) (id : code_id),
         (f, N, id) ∈ cl ->
         (f = slot ∧ N = Node_variable x)
         ∨ (f <> slot
            ∧ ∃ y : variable, lookup f others = Some y ∧ N = Node_variable y)
         ∨ (f <> slot
            ∧ ∃ k : positive,
                N = Node_made k ∧ (acc_next a <= k)%positive
                ∧ (k < acc_next a')%positive))
    ∧ (NoDup (map fst decls) ->
       (∀ (f : function_slot) (y : variable),
          lookup f others = Some y -> y <> x) ->
       (∀ (f f' : function_slot) (y : variable),
          lookup f others = Some y -> lookup f' others = Some y -> f = f') ->
       NoDup (map (fun '(_, N, _) => N) cl)).
Proof.
  induction decls as [ | [f id] decls IH]; intros a cl a' E;
    cbn [closures_of_set] in E.
  - injection E as <- <-.
    split; [ reflexivity | split; [ lia | split; [ intros f N id [] | ] ] ].
    intros _ _ _. constructor.
  - destruct (if Pos.eqb f slot then (Node_variable x, a)
              else match lookup f others with
                   | Some y => (Node_variable y, a)
                   | None => fresh a
                   end)
      as [name a1] eqn:E1.
    assert (H1 : (acc_next a <= acc_next a1)%positive
                 ∧ ((f = slot ∧ name = Node_variable x)
                    ∨ (f <> slot
                       ∧ ∃ y : variable,
                           lookup f others = Some y ∧ name = Node_variable y)
                    ∨ (f <> slot
                       ∧ lookup f others = None
                       ∧ name = Node_made (acc_next a)
                       ∧ acc_next a1 = Pos.succ (acc_next a)))).
    { destruct (Pos.eqb_spec f slot) as [Hfs | Hfs].
      - injection E1 as <- <-. split; [ lia | left; auto ].
      - destruct (lookup f others) as [y | ] eqn:Hy.
        + injection E1 as <- <-. split; [ lia | right; left ].
          split; [ exact Hfs | exists y; auto ].
        + injection E1 as <- <-. cbn. split; [ lia | right; right ].
          auto. }
    destruct H1 as (Hn1 & Hname).
    destruct (closures_of_set x slot others decls a1) as [rest a2] eqn:E2.
    injection E as <- <-.
    destruct (IH a1 rest a2 E2) as (Hmap & Hn2 & Hrest & Hnd).
    split; [ cbn [map]; rewrite Hmap; reflexivity | split; [ lia | split ] ].
    + intros g N id' [[= <- <- <-] | Hin].
      * destruct Hname as [? | [? | (Hfs & _ & -> & Hk)]]; auto.
        right. right. split; [ exact Hfs | ].
        exists (acc_next a). split; [ reflexivity | ]. lia.
      * destruct (Hrest g N id' Hin)
          as [? | [? | (Hgs & k & -> & Hk1 & Hk2)]]; auto.
        right. right. split; [ exact Hgs | ]. exists k.
        split; [ reflexivity | lia ].
    + intros Hdecls Hnx Hinj. cbn [map] in Hdecls |- *.
      inversion Hdecls as [ | f0 l0 Hf Hdecls' ]; subst.
      constructor; [ | exact (Hnd Hdecls' Hnx Hinj) ].
      intros Hin. apply in_map_iff in Hin as ([[g N] id'] & HN & Hin).
      cbn in HN. subst N.
      assert (Hg : g ∈ map fst decls).
      { rewrite <- Hmap. apply in_map_iff. exists (g, name, id'). auto. }
      assert (Hgf : g <> f) by (intros ->; exact (Hf Hg)).
      destruct (Hrest g name id' Hin)
        as [(-> & Hx) | [(_ & y & Hy & Hny) | (_ & k & Hk & Hk1 & _)]];
        destruct Hname
          as [(-> & Hx') | [(_ & y' & Hy' & Hny') | (_ & _ & Hk' & Hn)]].
      * exact (Hgf eq_refl).
      * rewrite Hx in Hny'. injection Hny' as <-.
        exact (Hnx f x Hy' eq_refl).
      * rewrite Hx in Hk'. discriminate Hk'.
      * rewrite Hny in Hx'. injection Hx' as ->.
        exact (Hnx g x Hy eq_refl).
      * rewrite Hny in Hny'. injection Hny' as <-.
        exact (Hgf (Hinj g f y Hy Hy')).
      * rewrite Hny in Hk'. discriminate Hk'.
      * rewrite Hk in Hx'. discriminate Hx'.
      * rewrite Hk in Hny'. discriminate Hny'.
      * rewrite Hk in Hk'. injection Hk' as ->. lia.
Qed.

Section Mutual.

Variable e0 : expr.
Hypothesis Hnl : NoDup (map fst (let_bindings e0)).

(* A function slot held by an allocation of the set of a [Let] of [z]:
   the allocations of the set ([map snd fs]) are distinct, include [z],
   are not symbols, and each holds every function slot of the set. *)
Definition let_call (a : acc) (x : node) (g : function_slot) (b : node)
    : Prop :=
  ∃ (z : variable) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (fs : list (function_slot * node)),
    (z, Closure_of_set slot decls vs) ∈ let_bindings e0
    ∧ (g, b) ∈ fs ∧ x ∈ map snd fs ∧ Node_variable z ∈ map snd fs
    ∧ NoDup (map snd fs) ∧ map fst fs = map fst (compact decls)
    ∧ (∀ (s : symbol) (y : node), y ∈ map snd fs -> y <> Node_symbol s)
    ∧ ∀ (y : node) (h : function_slot) (c : node),
        y ∈ map snd fs -> (h, c) ∈ fs ->
        Fact_constructor y (Field_function_slot h) c ∈ acc_facts a.

(* A function slot held by the symbol of a static closure: a symbol of its
   set. *)
Definition static_call (x b : node) (g : function_slot) : Prop :=
  ∃ (t : symbol) (slot : function_slot) (cs : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (sym : symbol),
    x = Node_symbol t ∧ b = Node_symbol sym
    ∧ (t, Static_const_closure_of_set slot cs decls vs) ∈ CE.sym_bindings e0
    ∧ (g, sym) ∈ compact cs.

Definition minv (a : acc) : Prop :=
  ∀ (x b : node) (g : function_slot),
    Fact_constructor x (Field_function_slot g) b ∈ acc_facts a ->
    let_call a x g b ∨ static_call x b g.

Lemma let_call_mono (a a' : acc) (x b : node) (g : function_slot) :
  (∀ fct : fact, fct ∈ acc_facts a -> fct ∈ acc_facts a') ->
  let_call a x g b -> let_call a' x g b.
Proof.
  intros Hsub (z & slot & decls & vs & fs & H1 & H2 & H3 & H4 & H5 & H6 & H7
               & H8).
  exists z, slot, decls, vs, fs. repeat (split; [ assumption | ]).
  intros y h c Hy Hhc. exact (Hsub _ (H8 y h c Hy Hhc)).
Qed.

Lemma minv_ns (a a' : acc) : minv a -> CE.ns_ok a a' -> minv a'.
Proof.
  intros Hm (H1 & H2 & _) x b g Hin.
  destruct (Hm x b g (H2 _ Hin eq_refl)) as [H | H];
    [ left; exact (let_call_mono a a' x b g H1 H) | right; exact H ].
Qed.

Lemma minv_record_plain (env : traverse_env)
    (closures : list (node * option code_id))
    (vs : list (value_slot * simple)) (a : acc) :
  minv a -> minv (record_set_of_closures_deps env closures [] vs a).
Proof.
  intros Hm.
  destruct (CE.record_set_facts env closures [] vs a) as [(L & HL & HLf) _].
  assert (Hsub : ∀ fct, fct ∈ acc_facts a ->
                   fct ∈ acc_facts (record_set_of_closures_deps env closures
                                      [] vs a))
    by (intros fct H; rewrite HL; apply in_or_app; right; exact H).
  intros x b g Hin. pose proof Hin as Hin0. rewrite HL in Hin0.
  apply in_app_or in Hin0 as [Hin0 | Hin0].
  - apply HLf in Hin0
      as (base & code & _ & [(w & s & _ & E) | (f & c & [] & _)]).
    discriminate E.
  - destruct (Hm x b g Hin0) as [H | H];
      [ left; exact (let_call_mono _ _ x b g Hsub H) | right; exact H ].
Qed.

Lemma minv_static (env : traverse_env) (s : symbol) (slot : function_slot)
    (cs : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (a : acc) :
  (s, Static_const_closure_of_set slot cs decls vs) ∈ CE.sym_bindings e0 ->
  minv a ->
  minv (traverse_static_binding env
          (Bind_symbol s (Static_const_closure_of_set slot cs decls vs)) a).
Proof.
  intros Hs Hm. cbn [traverse_static_binding].
  match goal with
  | |- minv (record_set_of_closures_deps env ?cl ?fs vs a) =>
    destruct (CE.record_set_facts env cl fs vs a) as [(L & HL & HLf) _]
  end.
  intros x b g Hin. pose proof Hin as Hin0. rewrite HL in Hin0.
  apply in_app_or in Hin0 as [Hin0 | Hin0].
  - right. apply HLf in Hin0
      as (base & code & Hbc & [(w & s0 & _ & E) | (f & c & Hfc & E)]);
      [ discriminate E | ].
    injection E as -> -> ->. destruct Hbc as [[= <- _] | []].
    apply in_map_iff in Hfc as ([f' sym] & Heq & Hfc). injection Heq as -> <-.
    exists s, slot, cs, decls, vs, sym. auto.
  - destruct (Hm x b g Hin0) as [H | H]; [ left | right; exact H ].
    apply (let_call_mono a _ x b g); [ | exact H ].
    intros fct Hf. rewrite HL. apply in_or_app. right. exact Hf.
Qed.

(* The [Let] of a closure adds the function slots of its set at each of
   its allocations. *)
Lemma minv_closure_let (env : traverse_env) (x : variable)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (a : acc) :
  env_set_closures env = set_closures e0 ->
  (x, Closure_of_set slot decls vs) ∈ let_bindings e0 ->
  minv a -> minv (traverse_closure_of_set env x slot decls vs a).
Proof.
  intros Henv Hx Hm. unfold traverse_closure_of_set.
  destruct (among slot (map fst (compact decls))) eqn:Hslot; [ | exact Hm ].
  destruct (closures_of_set x slot (set_closures_of env x) (compact decls) a)
    as [cl a1] eqn:Ecl.
  destruct (CE.closures_of_set_spec x slot _ _ a cl a1 Ecl) as (Hf1 & _).
  destruct (closures_of_set_exact x slot _ _ a cl a1 Ecl)
    as (Hmap & _ & Hcl & Hnd).
  set (closures := map (fun '(_, name, id) => (name, Some id)) cl).
  set (fs := map (fun '(f, name, _) => (f, name)) cl).
  destruct (CE.record_set_facts env closures fs vs a1) as [(L & HL & HLf) _].
  assert (Hsnd : map snd fs = map (fun '(_, N, _) => N) cl).
  { unfold fs. rewrite map_map. apply map_ext. intros [[f N] id].
    reflexivity. }
  assert (Hfst : map fst fs = map fst (compact decls)).
  { transitivity (map (fun '(f, _, _) => f) cl); [ | exact Hmap ].
    unfold fs. rewrite map_map. apply map_ext.
    intros [[f N] id]. reflexivity. }
  assert (Hbase : ∀ (y : node) (code : option code_id),
             (y, code) ∈ closures -> y ∈ map snd fs).
  { intros y code Hin. unfold closures in Hin.
    apply in_map_iff in Hin as ([[f N] id] & Heq & Hin).
    injection Heq as -> _. rewrite Hsnd. apply in_map_iff.
    exists (f, y, id). auto. }
  assert (Hcls : ∀ y : node, y ∈ map snd fs -> ∃ code, (y, code) ∈ closures).
  { intros y Hy. rewrite Hsnd in Hy.
    apply in_map_iff in Hy as ([[f N] id] & <- & Hin).
    exists (Some id). unfold closures. apply in_map_iff.
    exists (f, N, id). auto. }
  assert (Hothers :
    (∀ (f : function_slot) (y : variable),
       lookup f (set_closures_of env x) = Some y -> y <> x)
    ∧ ∀ (f f' : function_slot) (y : variable),
        lookup f (set_closures_of env x) = Some y ->
        lookup f' (set_closures_of env x) = Some y -> f = f').
  { unfold set_closures_of. rewrite Henv.
    destruct (lookup x (set_closures e0)) as [projs | ] eqn:Hp;
      [ | split; intros; discriminate ].
    pose proof (proj1 CE.set_closures_projs e0 x projs (lookup_in _ _ _ Hp))
      as Hproj.
    split.
    - intros f y Hy ->.
      destruct (Hproj f x (lookup_in _ _ _ Hy)) as (s & Hs).
      pose proof (CE.nodup_same _ _ _ _ Hnl Hx Hs) as E. discriminate E.
    - intros f f' y Hy Hy'.
      destruct (Hproj f y (lookup_in _ _ _ Hy)) as (s & Hs).
      destruct (Hproj f' y (lookup_in _ _ _ Hy')) as (s' & Hs').
      pose proof (CE.nodup_same _ _ _ _ Hnl Hs Hs') as E.
      injection E as _ E. exact E. }
  intros x' b g Hin. pose proof Hin as Hin0. rewrite HL in Hin0.
  apply in_app_or in Hin0 as [Hin0 | Hin0].
  - left. apply HLf in Hin0
      as (base & code & Hbc & [(w & s0 & _ & E) | (f & c & Hfc & E)]);
      [ discriminate E | ].
    injection E as -> -> ->.
    exists x, slot, decls, vs, fs.
    split; [ exact Hx | split; [ exact Hfc | ] ].
    split; [ exact (Hbase base code Hbc) | ].
    split.
    { apply among_In in Hslot.
      assert (Hslot' : slot ∈ map (fun '(f, _, _) => f) cl)
        by (rewrite Hmap; exact Hslot).
      apply in_map_iff in Hslot' as ([[f1 N] id] & Heq & Hin1). cbn in Heq.
      subst f1. rewrite Hsnd. apply in_map_iff. exists (slot, N, id).
      split; [ | exact Hin1 ].
      destruct (Hcl slot N id Hin1) as [(_ & ->) | [(Hne & _) | (Hne & _)]];
        [ reflexivity | destruct (Hne eq_refl) | destruct (Hne eq_refl) ]. }
    split.
    { rewrite Hsnd. apply (Hnd (CE.nodup_compact decls)); apply Hothers. }
    split; [ exact Hfst | split ].
    + intros s y Hy ->. rewrite Hsnd in Hy.
      apply in_map_iff in Hy as ([[f1 N] id] & HN & Hin1). cbn in HN.
      subst N. destruct (Hcl f1 _ id Hin1)
        as [(_ & E) | [(_ & y & _ & E) | (_ & k & E & _)]]; discriminate E.
    + intros y h c0 Hy Hhc. rewrite HL. apply in_or_app. left.
      apply HLf. destruct (Hcls y Hy) as (code0 & Hyc).
      exists y, code0. split; [ exact Hyc | right ]. exists h, c0. auto.
  - rewrite Hf1 in Hin0. destruct (Hm x' b g Hin0) as [H | H];
      [ left | right; exact H ].
    apply (let_call_mono a _ x' b g); [ | exact H ].
    intros fct Hf. rewrite HL, Hf1. apply in_or_app. right. exact Hf.
Qed.

Lemma minv_fold {A : Type} (f : acc -> A -> acc) (l : list A) :
  (∀ (a : acc) (y : A), y ∈ l -> minv a -> minv (f a y)) ->
  ∀ a : acc, minv a -> minv (List.fold_left f l a).
Proof.
  induction l as [ | y l IH]; intros Hf a Ha; [ exact Ha | ].
  cbn [List.fold_left]. apply IH; [ | apply Hf; [ left; reflexivity | ] ].
  - intros a' y' Hy'. apply Hf. right. exact Hy'.
  - exact Ha.
Qed.

Lemma set_closures_add_rec_conts (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) :
  env_set_closures (add_rec_conts env handlers) = env_set_closures env.
Proof.
  unfold add_rec_conts. induction handlers as [ | [k h] hs IH];
    [ reflexivity | exact IH ].
Qed.

(* The bindings of a part of [e0]. *)
Definition sub (L : list (variable * named))
    (S : list (symbol * static_const)) : Prop :=
  (∀ p, p ∈ L -> p ∈ let_bindings e0) ∧ (∀ p, p ∈ S -> p ∈ CE.sym_bindings e0).

Definition P_mut (e : expr) : Prop :=
  ∀ (env : traverse_env) (a : acc),
    sub (let_bindings e) (CE.sym_bindings e) ->
    env_set_closures env = set_closures e0 -> minv a ->
    minv (traverse env e a).

Lemma minv_traverse :
  (∀ e : expr, P_mut e)
  ∧ ∀ h : continuation_handler,
      match h with Handler _ _ handler => P_mut handler end.
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind; unfold P_mut.
  - intros x k d body IH env a [HL HS] Henv Hm.
    cbn [let_bindings CE.sym_bindings] in HL, HS. cbn [traverse].
    apply IH; [ split; [ intros p Hp; apply HL; right; exact Hp | exact HS ]
              | exact Henv | ].
    destruct d as [s | p args | id vs | slot decls vs | w c | pf w c
                  | mf mt c];
      try (apply (minv_ns a); [ exact Hm | ];
           apply CE.ns_traverse_named; intros; discriminate).
    + cbn [traverse_named]. apply minv_record_plain. exact Hm.
    + cbn [traverse_named]. apply minv_closure_let;
        [ exact Henv | apply HL; left; reflexivity | exact Hm ].
  - intros bindings body Hcodes IH env a [HL HS] Henv Hm.
    cbn [let_bindings CE.sym_bindings] in HL, HS. cbn [traverse].
    apply IH;
      [ split; intros p Hp; [ apply HL | apply HS ]; apply in_or_app; right;
        exact Hp
      | exact Henv | ].
    apply minv_fold; [ | apply minv_fold; [ | apply minv_fold ] ].
    + intros a' [id c | id | s const] Hb Hm'; cbn [traverse_binding];
        [ | exact Hm' | exact Hm' ].
      destruct c as [params mc ret exn md cbody]. cbn [traverse_code].
      destruct (lookup id (acc_code_deps a')) as [cd | ]; [ | exact Hm' ].
      pose proof (Hcodes id (Code params mc ret exn md cbody) Hb) as Hc.
      cbn in Hc. apply Hc; [ | exact Henv | ].
      * split; intros p Hp; [ apply HL | apply HS ]; apply in_or_app; left;
          apply in_flat_map; exists (Bind_code id (Code params mc ret exn md
            cbody)); split; [ exact Hb | exact Hp | exact Hb | exact Hp ].
      * apply (minv_ns a'); [ exact Hm' | apply CE.ns_code_aliases ].
    + intros a' b Hb Hm'.
      destruct b as [id c | id | s const];
        [ exact Hm' | exact Hm' | ].
      destruct const as [id vs | slot cs decls vs | | | | ];
        try (apply (minv_ns a'); [ exact Hm' | ];
             apply CE.ns_traverse_static_binding; intros; discriminate).
      * cbn [traverse_static_binding]. apply minv_record_plain. exact Hm'.
      * apply minv_static; [ | exact Hm' ].
        apply HS. apply in_or_app. left. apply in_flat_map.
        exists (Bind_symbol s (Static_const_closure_of_set slot cs decls vs)).
        split; [ exact Hb | left; reflexivity ].
    + intros a' b _ Hm'. apply (minv_ns a'); [ exact Hm' | ].
      apply CE.ns_prepare_binding.
    + exact Hm.
  - intros k h body IHh IH env a [HL HS] Henv Hm.
    destruct h as [params md handler].
    cbn [let_bindings CE.sym_bindings handler_let_bindings
         CE.handler_sym_bindings] in HL, HS.
    cbn [traverse traverse_handler] in IHh |- *.
    apply IH;
      [ split; intros p Hp; [ apply HL | apply HS ]; apply in_or_app; right;
        exact Hp
      | exact Henv | ].
    apply (minv_ns (traverse env handler a));
      [ | apply CE.ns_exn_handler_deps ].
    apply IHh; [ | exact Henv | exact Hm ].
    split; intros p Hp; [ apply HL | apply HS ]; apply in_or_app; left;
      exact Hp.
  - intros handlers body IHh IH env a [HL HS] Henv Hm.
    cbn [let_bindings CE.sym_bindings] in HL, HS. cbn [traverse].
    assert (Henv' : env_set_closures (add_rec_conts env handlers)
                    = set_closures e0)
      by (rewrite set_closures_add_rec_conts; exact Henv).
    apply IH;
      [ split; intros p Hp; [ apply HL | apply HS ]; apply in_or_app; right;
        exact Hp
      | exact Henv' | ].
    apply minv_fold; [ | exact Hm ].
    intros a' [k h] Hkh Hm'. destruct h as [params md handler].
    pose proof (IHh k _ Hkh) as Hh. cbn [traverse_handler] in Hh |- *.
    apply Hh; [ | exact Henv' | exact Hm' ].
    split; intros p Hp; [ apply HL | apply HS ]; apply in_or_app; left;
      apply in_flat_map; exists (k, Handler params md handler);
      (split; [ exact Hkh | exact Hp ]).
  - intros ap env a _ _ Hm. apply (minv_ns a); [ exact Hm | ].
    apply CE.ns_traverse_apply.
  - intros ac env a _ _ Hm. apply (minv_ns a); [ exact Hm | ].
    apply CE.ns_apply_cont_deps.
  - intros scrutinee arms env a _ _ Hm. apply (minv_ns a); [ exact Hm | ].
    apply CE.ns_traverse_switch.
  - intros env a _ _ Hm. exact Hm.
  - intros params md handler IH. exact IH.
Qed.

End Mutual.

(* The traversal of a unit keeps [minv]. *)
Lemma minv_unit (flag : reaper_preserve_direct_calls) (u : unit_interface)
    (e : expr) :
  NoDup (map fst (let_bindings e)) ->
  ∃ a : acc, traverse_unit flag u e = facts_of_list (acc_facts a) ∧ minv e a.
Proof.
  intros Hnl. unfold traverse_unit.
  set (a0 := add_fact (Fact_any_source Node_all_constants) empty_acc).
  destruct (fresh a0) as [r a1] eqn:E1.
  destruct (fresh a1) as [x a2] eqn:E2.
  set (a3 := add_fact (Fact_any_usage x) a2).
  set (env := {| env_conts := [(unit_exn u, [x]); (unit_return u, [r])];
                 env_current_code_id := None;
                 env_should_preserve_direct_calls :=
                   should_preserve_of_flag flag;
                 env_value_slots_to_keep := projected_value_slots e;
                 env_unit_code_ids := static_names all_code_ids_of_binding e;
                 env_set_closures := set_closures e |}).
  assert (Hm3 : minv e a3).
  { assert (Hf : acc_facts a3
                 = [Fact_any_usage x; Fact_any_source Node_all_constants]).
    { unfold a3. cbn [add_fact add_facts acc_facts app].
      assert (Hf2 : acc_facts a2 = acc_facts a1)
        by (change a2 with (snd (x, a2)); rewrite <- E2; reflexivity).
      assert (Hf1 : acc_facts a1 = acc_facts a0)
        by (change a1 with (snd (r, a1)); rewrite <- E1; reflexivity).
      rewrite Hf2, Hf1. reflexivity. }
    intros y b g Hin. rewrite Hf in Hin.
    destruct Hin as [E | [E | []]]; discriminate E. }
  eexists. split; [ reflexivity | ].
  apply (minv_ns e (add_fact (Fact_any_usage r) (traverse env e a3)));
    [ | apply CE.ns_resolve ].
  apply (minv_ns e (traverse env e a3));
    [ | apply CE.ns_add_fact; cbn; auto ].
  apply (proj1 (minv_traverse e Hnl) e env a3);
    [ split; auto | reflexivity | exact Hm3 ].
Qed.

(* ======================================================================== *)
(* The graph of the traversal                                               *)
(* ======================================================================== *)

Lemma nodup_snd_same {A B : Type} (l : list (A * B)) (x y : A) (c : B) :
  NoDup (map snd l) -> (x, c) ∈ l -> (y, c) ∈ l -> x = y.
Proof.
  induction l as [ | [x' c'] l IH]; intros Hnd Hx Hy; [ destruct Hx | ].
  inversion Hnd as [ | c0 l0 Hc Hnd' ]; subst.
  destruct Hx as [Hx | Hx]; destruct Hy as [Hy | Hy].
  - congruence.
  - injection Hx as -> ->. exfalso. apply Hc.
    exact (in_map snd l (y, c) Hy).
  - injection Hy as -> ->. exfalso. apply Hc.
    exact (in_map snd l (x, c) Hx).
  - exact (IH Hnd' Hx Hy).
Qed.

(* What the premises of the statements give of the unit. *)
Lemma unit_premises (u : unit_interface) (e : expr) :
  in_flow_fragment e = true -> well_formed_unit u e = true ->
  variables_bound_once e = true ->
  CE.no_old_sets e ∧ NoDup (map fst (let_bindings e))
  ∧ NoDup (map fst (CE.sym_bindings e)) ∧ sets_well_formed e = true.
Proof.
  intros Hfrag Hwf Honce.
  assert (Hold : CE.no_old_sets e).
  { unfold in_flow_fragment in Hfrag. apply andb_true_iff in Hfrag as [Hf _].
    exact (proj1 (CE.fragment_no_old_sets _ _ _) e Hf). }
  split; [ exact Hold | split; [ | split ] ].
  - unfold variables_bound_once in Honce. apply distinct_NoDup in Honce.
    exact (NoDup_app_remove_r _ _ Honce).
  - rewrite (proj1 CE.sym_bindings_static_names e Hold).
    apply distinct_NoDup. unfold well_formed_unit in Hwf.
    repeat (apply andb_true_iff in Hwf as [Hwf ?]).
    assumption.
  - unfold well_formed_unit in Hwf.
    repeat (apply andb_true_iff in Hwf as [Hwf ?]).
    assumption.
Qed.

Section Graph.

Variable flag : reaper_preserve_direct_calls.
Variable u : unit_interface.
Variable e : expr.
Hypothesis Hnl : NoDup (map fst (let_bindings e)).
Hypothesis Hns : NoDup (map fst (CE.sym_bindings e)).
Hypothesis Hwf : sets_well_formed e = true.
Variable Gs : list CE.sgroup.
Variable a : acc.
Hypothesis Ha : traverse_unit flag u e = facts_of_list (acc_facts a).
Hypothesis Hinv : CE.inv e (unit_kept e) Gs a.
Hypothesis Hpres :
  ∀ N : node, CE.static_origin_in e N ->
    ∃ γ : CE.sgroup, γ ∈ Gs ∧ CE.sg_origin γ = N.
Variable a' : acc.
Hypothesis Ha' : traverse_unit flag u e = facts_of_list (acc_facts a').
Hypothesis Hm : minv e a'.

Abbreviation G := (graph_of_facts (traverse_unit flag u e)).
Abbreviation fsf x g b := (g_constructor G x (Field_function_slot g) b).

Lemma fsf_minv (x b : node) (g : function_slot) :
  fsf x g b <-> Fact_constructor x (Field_function_slot g) b ∈ acc_facts a'.
Proof. rewrite Ha'. apply CE.constructor_of_list. Qed.

(* The group of an allocation: its function slots are those of the
   group. *)
Lemma group_of (x b : node) (g : function_slot) :
  fsf x g b ->
  ∃ γ : CE.sgroup,
    γ ∈ Gs ∧ x ∈ CE.sg_bases γ
    ∧ ∀ (h : function_slot) (c : node),
        fsf x h c <-> (h, c) ∈ CE.sg_fslots γ.
Proof.
  rewrite Ha. intros H.
  destruct (CE.group_at e Gs a Hinv x b _ H eq_refl) as (γ & Hγ & Hx).
  exists γ. split; [ exact Hγ | split; [ exact Hx | ] ].
  exact (proj1 (CE.facts_at e Gs a Hinv γ x Hγ Hx)).
Qed.

(* The function slots of a symbol of a static set are those of the set. *)
Lemma static_group (t sym : symbol) (g slot : function_slot)
    (cs : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) :
  (t, Static_const_closure_of_set slot cs decls vs) ∈ CE.sym_bindings e ->
  (g, sym) ∈ compact cs ->
  ∀ (h : function_slot) (c : node),
    fsf (Node_symbol sym) h c
    <-> (h, c) ∈ map (fun '(f, s) => (f, Node_symbol s)) (compact cs).
Proof.
  intros Hs Hin.
  apply in_compact in Hin. apply lookup_in in Hin.
  destruct (proj1 CE.static_sets_whole e Hwf t slot cs decls vs Hs)
    as [_ Hwhole].
  pose proof (Hwhole g sym Hin) as Hsym.
  destruct (Hpres (Node_symbol sym)) as (γ & Hγ & Ho).
  { exists sym, g, cs, decls, vs. auto. }
  destruct Hinv as (_ & _ & _ & HD & _).
  destruct (HD γ Hγ)
    as (_ & _ & _ & [(x & _ & _ & _ & Ox & _)
                    | (t' & _ & _ & _ & _ & Ot & _ & Hb & _)]);
    [ rewrite Ox in Ho; discriminate Ho | ].
  rewrite Ot in Ho. injection Ho as ->.
  assert (Hbs : Node_symbol sym ∈ CE.sg_bases γ)
    by (rewrite Hb; left; reflexivity).
  pose proof (CE.static_symbol_group e Hns Gs a Hinv γ sym g cs decls vs Hγ
                Hbs Hsym) as [Hf _].
  intros h c. rewrite Ha, <- Hf.
  exact (proj1 (CE.facts_at e Gs a Hinv γ _ Hγ Hbs) h c).
Qed.

Lemma graph_functional (x b b' : node) (g : function_slot) :
  fsf x g b -> fsf x g b' -> b = b'.
Proof.
  intros H H'. destruct (group_of x b g H) as (γ & Hγ & _ & Hf).
  exact (CE.nodup_same _ g b b' (proj1 (CE.group_nodup e Gs a Hinv γ Hγ))
           (proj1 (Hf g b) H) (proj1 (Hf g b') H')).
Qed.

Lemma graph_closed (x b c : node) (g h : function_slot) :
  fsf x g b -> fsf x h c <-> fsf b h c.
Proof.
  intros H. destruct (group_of x b g H) as (γ & Hγ & Hx & Hf).
  pose proof (proj1 (Hf g b) H) as Hgb.
  rewrite (Hf h c).
  destruct Hinv as (_ & _ & _ & HD & _).
  destruct (HD γ Hγ)
    as (_ & _ & _ & [(z & slot & decls & vs & _ & _ & _ & _ & Htgt & _)
                    | (t & slot & cs & decls & vs & _ & Hs & Hb & Hfs & _)]).
  - pose proof (Htgt g b Hgb) as Hbb.
    rewrite Ha. symmetry.
    exact (proj1 (CE.facts_at e Gs a Hinv γ b Hγ Hbb) h c).
  - rewrite Hfs in Hgb |- *.
    apply in_map_iff in Hgb as ([g' sym] & Heq & Hin). injection Heq as -> <-.
    symmetry. exact (static_group t sym g slot cs decls vs Hs Hin h c).
Qed.

Lemma graph_injective (x b : node) (g g' : function_slot) :
  fsf x g b -> fsf x g' b -> g = g'.
Proof.
  intros H H'. destruct (group_of x b g H) as (γ & Hγ & Hx & Hf).
  pose proof (proj1 (Hf g b) H) as Hgb.
  pose proof (proj1 (Hf g' b) H') as Hgb'.
  destruct Hinv as (_ & _ & HC & HD & _).
  destruct (HD γ Hγ)
    as (_ & _ & _ & [(z & slot & decls & vs & _ & Hlet & _ & Hdom & Htgt
                      & Hbases)
                    | (t & slot & cs & decls & vs & _ & Hs & Hb & Hfs & _)]).
  - destruct (Hm x b g (proj1 (fsf_minv x b g) H))
      as [(z1 & slot1 & decls1 & vs1 & fs1 & Hz1 & Hgb1 & Hxfs & Hzfs & Hnd1
           & Hfst1 & _ & Hfacts)
         | (t & _ & _ & _ & _ & _ & -> & _ & _ & _)].
    2:{ destruct (Hbases _ Hx) as [E | [(y & s & o & E & _) | (k & E)]];
        discriminate E. }
    (* The [Let] of the group is that of [z1]. *)
    assert (Hz1b : fsf (Node_variable z1) g b)
      by exact (proj2 (fsf_minv _ b g) (Hfacts _ g b Hzfs Hgb1)).
    destruct (group_of _ b g Hz1b) as (γ1 & Hγ1 & Hz1in & Hf1).
    assert (Eγ : γ1 = γ).
    { destruct (HD γ1 Hγ1)
        as (_ & _ & _ & [(z' & _ & _ & _ & _ & _ & _ & _ & Htgt1 & _)
                        | (t1 & _ & _ & _ & _ & _ & _ & Hb1 & _)]).
      - apply (HC γ1 γ b Hγ1 Hγ); [ apply (Htgt1 g b) | apply (Htgt g b) ];
          [ exact (proj1 (Hf1 g b) Hz1b) | exact Hgb ].
      - rewrite Hb1 in Hz1in. destruct Hz1in as [E | []]. discriminate E. }
    subst γ1.
    assert (Ez : z1 = z).
    { destruct (Hbases _ Hz1in)
        as [[= E] | [(y & s & o & [= <-] & Hy) | (k & E)]];
        [ exact E | | discriminate E ].
      pose proof (CE.nodup_same _ _ _ _ Hnl Hz1 Hy) as E. discriminate E. }
    subst z1.
    pose proof (CE.nodup_same _ _ _ _ Hnl Hz1 Hlet) as E.
    injection E as _ <- _.
    assert (Hin1 : g' ∈ map fst (compact decls1)).
    { apply among_In. rewrite <- is_bound_among, CE.is_bound_compact, <- Hdom.
      exact (CE.in_is_bound g' b _ Hgb'). }
    assert (Hg'1 : g' ∈ map fst fs1) by (rewrite Hfst1; exact Hin1).
    apply in_map_iff in Hg'1 as ([g'' c] & Heq & Hc). cbn in Heq. subst g''.
    pose proof (proj2 (fsf_minv x c g') (Hfacts x g' c Hxfs Hc)) as Hxc.
    rewrite (graph_functional x c b g' Hxc H') in Hc.
    exact (nodup_snd_same fs1 g g' b Hnd1 Hgb1 Hc).
  - rewrite Hfs in Hgb, Hgb'.
    apply in_map_iff in Hgb as ([g1 sym] & Heq & Hin). injection Heq as -> <-.
    apply in_map_iff in Hgb' as ([g2 sym'] & Heq & Hin').
    injection Heq as -> ->.
    apply in_compact in Hin, Hin'. apply lookup_in in Hin, Hin'.
    destruct (proj1 CE.static_sets_whole e Hwf t slot cs decls vs Hs)
      as [_ Hwhole].
    pose proof (CE.nodup_same _ _ _ _ Hns (Hwhole _ _ Hin) (Hwhole _ _ Hin'))
      as E.
    injection E as E. exact E.
Qed.

Lemma graph_mutual (x b : node) (g : function_slot) :
  fsf x g b -> ∃ g' : function_slot, fsf b g' x.
Proof.
  intros H.
  destruct (Hm x b g (proj1 (fsf_minv x b g) H))
    as [(z & slot & decls & vs & fs & _ & Hgb & Hxfs & _ & _ & _ & _
         & Hfacts)
       | (t & slot & cs & decls & vs & sym & -> & -> & Hs & Hin)].
  - apply in_map_iff in Hxfs as ([g' x'] & Heq & Hgx). cbn in Heq. subst x'.
    exists g'. apply (proj2 (fsf_minv b x g')).
    apply (Hfacts b g' x); [ | exact Hgx ].
    exact (in_map snd fs (g, b) Hgb).
  - exists slot.
    apply (proj2 (static_group t sym g slot cs decls vs Hs Hin slot
                    (Node_symbol t))).
    apply in_map_iff. exists (slot, t). split; [ reflexivity | ].
    apply in_compact. exact (static_lookup e t slot cs decls vs Hwf Hs).
Qed.

Lemma graph_props :
  function_slots_mutual G ∧ function_slots_of_sets G.
Proof.
  split; [ exact graph_mutual | ].
  split; [ exact graph_functional | split; [ exact graph_injective | ] ].
  exact graph_closed.
Qed.

End Graph.

Lemma traversal_props (u : unit_interface) (e : expr) :
  in_flow_fragment e = true -> well_formed_unit u e = true ->
  variables_bound_once e = true ->
  function_slots_mutual (graph_of_facts (traverse_unit Auto u e))
  ∧ function_slots_of_sets (graph_of_facts (traverse_unit Auto u e)).
Proof.
  intros Hfrag Hwfu Honce.
  destruct (unit_premises u e Hfrag Hwfu Honce) as (Hold & Hnl & Hns & Hwf).
  destruct (CE.traverse_unit_groups Auto u e Hnl Hns Hold)
    as (Gs & a & Ha & Hinv & Hpres).
  destruct (minv_unit Auto u e Hnl) as (a' & Ha' & Hm).
  exact (graph_props Auto u e Hnl Hns Hwf Gs a Ha Hinv Hpres a' Ha' Hm).
Qed.

Theorem function_slots_mutual_traversal :
  function_slots_mutual_traversal_statement.
Proof.
  intros u e Hfrag Hwfu Honce. exact (proj1 (traversal_props u e Hfrag Hwfu
                                                Honce)).
Qed.

Theorem function_slots_of_sets_traversal :
  function_slots_of_sets_traversal_statement.
Proof.
  intros u e Hfrag Hwfu Honce. exact (proj2 (traversal_props u e Hfrag Hwfu
                                                Honce)).
Qed.

(* ======================================================================== *)
(* The kinds of the fields of blocks                                        *)
(* ======================================================================== *)

Module Kinds.

(* A constructor fact at a field of a block of a kind other than value. *)
Definition bad_fact (fct : fact) : Prop :=
  match fct with
  | Fact_constructor _ (Field_block _ Value) _ => False
  | Fact_constructor _ (Field_block _ _) _ => True
  | _ => False
  end.

(* From [a] to [a'], facts are added that are not bad facts, and nodes
   are made: the [ns_ok] of FlambdaReaperClosuresExactProofs.v for
   [bad_fact]. *)
Definition ko_ok (a a' : acc) : Prop :=
  (∀ fct : fact, fct ∈ acc_facts a -> fct ∈ acc_facts a')
  ∧ (∀ fct : fact,
       fct ∈ acc_facts a' -> bad_fact fct -> fct ∈ acc_facts a)
  ∧ (acc_next a <= acc_next a')%positive.

Lemma ko_refl (a : acc) : ko_ok a a.
Proof. split; [ auto | split; [ auto | lia ] ]. Qed.

Lemma ko_trans (a a1 a2 : acc) : ko_ok a a1 -> ko_ok a1 a2 -> ko_ok a a2.
Proof.
  intros (H1 & H2 & H3) (H1' & H2' & H3').
  split; [ auto | split; [ auto | lia ] ].
Qed.

Lemma ko_same (a a' : acc) :
  acc_facts a' = acc_facts a -> (acc_next a <= acc_next a')%positive ->
  ko_ok a a'.
Proof.
  intros E Hn. split; [ rewrite E; auto | split; [ rewrite E; auto | lia ] ].
Qed.

Lemma ko_add_facts (L : list fact) (a : acc) :
  (∀ fct : fact, fct ∈ L -> ~ bad_fact fct) -> ko_ok a (add_facts L a).
Proof.
  intros HL. unfold ko_ok. cbn [add_facts acc_facts acc_next].
  split; [ intros fct Hin; apply in_or_app; right; exact Hin | ].
  split; [ | lia ].
  intros fct Hin Hs. apply in_app_or in Hin as [Hin | Hin];
    [ destruct (HL fct Hin Hs) | exact Hin ].
Qed.

Lemma ko_add_fact (fct : fact) (a : acc) :
  ~ bad_fact fct -> ko_ok a (add_fact fct a).
Proof.
  intros H. apply ko_add_facts. intros fct' [<- | []]. exact H.
Qed.

(* Shows that the facts of a list are not bad facts. *)
Ltac bad_free :=
  let fct := fresh "fct" in let Hin := fresh "Hin" in
  let Hs := fresh "Hs" in
  intros fct Hin Hs;
  repeat match goal with
  | H : In _ (_ ++ _) |- _ => apply in_app_or in H; destruct H as [H | H]
  | H : In _ (map _ _) |- _ =>
    let y := fresh "y" in let Heq := fresh "Heq" in
    apply in_map_iff in H; destruct H as [y [Heq _]]; subst fct;
    repeat match goal with
           | p : (_ * _)%type |- _ => destruct p
           end
  | H : In _ (_ :: _) |- _ => destruct H as [H | H]; [ subst fct | ]
  | H : In _ [] |- _ => destruct H
  end;
  cbn [bad_fact] in *; try discriminate; try contradiction.

Lemma ko_fresh (a : acc) : ko_ok a (snd (fresh a)).
Proof. apply ko_same; cbn; [ reflexivity | lia ]. Qed.

Lemma ko_fresh_list (n : nat) : ∀ a : acc, ko_ok a (snd (fresh_list n a)).
Proof.
  induction n as [ | n IH]; intros a; [ apply ko_refl | ].
  cbn [fresh_list]. destruct (fresh a) as [x a1] eqn:E1.
  destruct (fresh_list n a1) as [xs a2] eqn:E2. cbn [snd].
  apply (ko_trans _ a1).
  - replace a1 with (snd (fresh a)) by (rewrite E1; reflexivity).
    apply ko_fresh.
  - replace a2 with (snd (fresh_list n a1)) by (rewrite E2; reflexivity).
    apply IH.
Qed.

Lemma ko_note_simple (s : simple) (a : acc) : ko_ok a (note_simple s a).
Proof. destruct s; apply ko_same; cbn; (reflexivity || lia). Qed.

Lemma ko_fold {A : Type} (f : acc -> A -> acc) (l : list A) :
  (∀ (a : acc) (x : A), x ∈ l -> ko_ok a (f a x)) ->
  ∀ a : acc, ko_ok a (List.fold_left f l a).
Proof.
  induction l as [ | x l IH]; intros Hf a; [ apply ko_refl | ].
  cbn [List.fold_left].
  apply (ko_trans _ (f a x)); [ apply Hf; left; reflexivity | ].
  apply IH. intros a' y Hy. apply Hf. right. exact Hy.
Qed.

Lemma ko_note_simples (l : list simple) (a : acc) :
  ko_ok a (note_simples l a).
Proof. apply ko_fold. intros a' s _. apply ko_note_simple. Qed.

Lemma ko_push_apply_dep (d : apply_dep) (a : acc) :
  ko_ok a (push_apply_dep d a).
Proof. apply ko_same; cbn; (reflexivity || lia). Qed.

Lemma ko_push_closure_dep (d : closure_dep) (a : acc) :
  ko_ok a (push_closure_dep d a).
Proof. apply ko_same; cbn; (reflexivity || lia). Qed.

Lemma ko_add_code_dep (id : code_id) (cd : code_dep) (a : acc) :
  ko_ok a (add_code_dep id cd a).
Proof. apply ko_same; cbn; (reflexivity || lia). Qed.

Lemma ko_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  ko_ok a (cond_any_usage env s a).
Proof.
  unfold cond_any_usage. apply (ko_trans _ (note_simple s a));
    [ apply ko_note_simple | ].
  destruct (env_current_code_id env); apply ko_add_fact; cbn; auto.
Qed.

Lemma ko_cond_alias (env : traverse_env) (from to_ : node) (a : acc) :
  ko_ok a (cond_alias env from to_ a).
Proof.
  unfold cond_alias.
  destruct (env_current_code_id env); apply ko_add_fact; cbn; auto.
Qed.

(* Steps whose result is a node and an accumulator. *)
Ltac ko_let H lem :=
  match type of H with
  | ?f = (?x, ?a1) =>
    let Hn := fresh "Hns" in
    assert (Hn : ko_ok _ a1)
      by (change a1 with (snd (x, a1)); rewrite <- H; apply lem)
  end.

(* ======================================================================== *)
(* The steps                                                                *)
(* ======================================================================== *)

Lemma ko_find_cont (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  ko_ok a (snd (find_cont env k n a)).
Proof.
  unfold find_cont. destruct (lookup k (env_conts env)); [ apply ko_refl | ].
  destruct (fresh_list n a) as [params a1] eqn:E. ko_let E ko_fresh_list.
  apply (ko_trans _ _ _ Hns). apply ko_add_facts. bad_free.
Qed.

Lemma ko_block_deps (base : node) (fields : list simple) (a : acc) :
  ko_ok a (block_deps base fields a).
Proof.
  unfold block_deps. apply (ko_trans _ _ _ (ko_note_simples fields a)).
  apply ko_add_facts. bad_free.
Qed.

Lemma ko_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  ko_ok a (traverse_prim env x p args a).
Proof.
  assert (Hacc : ∀ (f : field) (s : simple) (a0 : acc),
             ko_ok a0 (add_fact (Fact_accessor (Node_variable x) f
                                   (node_of_simple s)) (note_simple s a0))).
  { intros f s a0. apply (ko_trans _ _ _ (ko_note_simple s a0)).
    apply ko_add_fact. cbn. auto. }
  assert (Hdefault :
            ko_ok a
              (add_facts
                 (Fact_any_source (Node_variable x)
                  :: map (fun from => Fact_use (Node_variable x) from)
                       (free_names_of_simples args))
                 (if prim_has_arbitrary_effects p
                  then cond_any_usage env (Var x) a else a))).
  { apply (ko_trans _ (if prim_has_arbitrary_effects p
                        then cond_any_usage env (Var x) a else a)).
    - destruct (prim_has_arbitrary_effects p);
        [ apply ko_cond_any_usage | apply ko_refl ].
    - apply ko_add_facts. bad_free. }
  unfold traverse_prim; cbv zeta.
  destruct p as [u0 | b0 | v0 | h0 | r0];
    [ destruct u0 | | destruct v0 | destruct h0 | destruct r0 ];
    try (destruct args as [ | arg [ | arg2 args]]);
    cbv beta iota;
    try first [ exact Hdefault | apply ko_block_deps | apply ko_refl
              | exact (Hacc _ _ _) ].
  all: try (destruct variant_only;
            first [ exact Hdefault | exact (Hacc _ _ _) ]).
  - destruct mut; [ | exact (Hacc _ _ _) | exact (Hacc _ _ _) ].
    eapply ko_trans; [ | apply ko_add_fact; cbn; auto ]. apply Hacc.
  - apply (ko_trans _ _ _ (ko_note_simple arg a)).
    apply ko_add_fact. cbn. auto.
Qed.

Lemma ko_create_known (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  ko_ok a (snd (create_known_arity_call_witness id params returns exn a)).
Proof.
  unfold create_known_arity_call_witness.
  destruct (fresh a) as [w a1] eqn:E. ko_let E ko_fresh.
  apply (ko_trans _ _ _ Hns). apply ko_add_facts. bad_free.
Qed.

Lemma ko_witness_deps (id : code_id) (returns : list node) (exn : node) :
  ∀ (l : list (list node * node)) (a : acc),
    ko_ok a (unknown_arity_witness_deps id returns exn l a).
Proof.
  induction l as [ | [first w] rest IH]; intros a; [ apply ko_refl | ].
  cbn [unknown_arity_witness_deps].
  match goal with
  | |- context [add_facts ?L a] =>
    apply (ko_trans _ (add_facts L a)); [ apply ko_add_facts; bad_free | ]
  end.
  destruct rest as [ | [? next] rest'].
  - apply ko_add_facts. bad_free.
  - match goal with
    | |- context [fresh ?b] => destruct (fresh b) as [v a1] eqn:E
    end.
    ko_let E ko_fresh. apply (ko_trans _ _ _ Hns).
    match goal with
    | |- context [add_facts ?L a1] =>
      apply (ko_trans _ (add_facts L a1)); [ apply ko_add_facts; bad_free | ]
    end.
    apply IH.
Qed.

Lemma ko_create_unknown (id : code_id) (is_tupled : bool) (ar : arity)
    (params returns : list node) (exn : node) (a : acc) :
  ko_ok a
    (snd (create_unknown_arity_call_witnesses id is_tupled ar params returns
            exn a)).
Proof.
  unfold create_unknown_arity_call_witnesses. destruct is_tupled.
  - unfold create_unknown_arity_tupled_call_witnesses.
    destruct (fresh a) as [w a1] eqn:E1. ko_let E1 ko_fresh.
    destruct (fresh a1) as [u a2] eqn:E2.
    assert (Hns2 : ko_ok a1 a2)
      by (change a2 with (snd (u, a2)); rewrite <- E2; apply ko_fresh).
    destruct (fresh a2) as [k a3] eqn:E3.
    assert (Hns3 : ko_ok a2 a3)
      by (change a3 with (snd (k, a3)); rewrite <- E3; apply ko_fresh).
    cbn [snd]. apply (ko_trans _ _ _ Hns), (ko_trans _ _ _ Hns2),
      (ko_trans _ _ _ Hns3).
    apply ko_add_facts. bad_free.
  - unfold create_unknown_arity_non_tupled_call_witnesses.
    match goal with
    | |- context [fresh_list ?n a] => destruct (fresh_list n a) as [ws a1] eqn:E
    end.
    ko_let E ko_fresh_list. cbn [snd]. apply (ko_trans _ _ _ Hns).
    apply ko_witness_deps.
Qed.

Lemma ko_prepare_code (id : code_id) (c : code) (a : acc) :
  ko_ok a (prepare_code id c a).
Proof.
  unfold prepare_code.
  destruct (fresh_list _ a) as [returns a1] eqn:E1. ko_let E1 ko_fresh_list.
  destruct (fresh a1) as [exn a2] eqn:E2.
  assert (Hns2 : ko_ok a1 a2)
    by (change a2 with (snd (exn, a2)); rewrite <- E2; apply ko_fresh).
  destruct (fresh a2) as [mc a3] eqn:E3.
  assert (Hns3 : ko_ok a2 a3)
    by (change a3 with (snd (mc, a3)); rewrite <- E3; apply ko_fresh).
  destruct (fresh_list _ a3) as [params a4] eqn:E4.
  assert (Hns4 : ko_ok a3 a4)
    by (change a4 with (snd (params, a4)); rewrite <- E4;
        apply ko_fresh_list).
  destruct (create_known_arity_call_witness id params returns exn a4)
    as [known a5] eqn:E5.
  assert (Hns5 : ko_ok a4 a5)
    by (change a5 with (snd (known, a5)); rewrite <- E5;
        apply ko_create_known).
  destruct (create_unknown_arity_call_witnesses _ _ _ _ _ _ a5)
    as [unknown a6] eqn:E6.
  assert (Hns6 : ko_ok a5 a6)
    by (change a6 with (snd (unknown, a6)); rewrite <- E6;
        apply ko_create_unknown).
  apply (ko_trans _ _ _ Hns), (ko_trans _ _ _ Hns2), (ko_trans _ _ _ Hns3),
    (ko_trans _ _ _ Hns4), (ko_trans _ _ _ Hns5), (ko_trans _ _ _ Hns6).
  match goal with
  | |- context [add_fact ?f a6] =>
    apply (ko_trans _ (add_fact f a6)); [ apply ko_add_fact; cbn; auto | ]
  end.
  apply ko_add_code_dep.
Qed.

Lemma ko_prepare_binding (b : static_binding) (a : acc) :
  ko_ok a (prepare_binding b a).
Proof.
  destruct b; [ apply ko_prepare_code | apply ko_refl | apply ko_refl ].
Qed.

Lemma ko_make_known (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  ko_ok a (snd (make_known_arity_apply_widget env ap returns exn a)).
Proof.
  unfold make_known_arity_apply_widget.
  destruct (fresh a) as [w a1] eqn:E1. ko_let E1 ko_fresh.
  destruct (fresh a1) as [called a2] eqn:E2.
  assert (Hns2 : ko_ok a1 a2)
    by (change a2 with (snd (called, a2)); rewrite <- E2; apply ko_fresh).
  destruct (fresh a2) as [an a3] eqn:E3.
  assert (Hns3 : ko_ok a2 a3)
    by (change a3 with (snd (an, a3)); rewrite <- E3; apply ko_fresh).
  cbn [snd].
  apply (ko_trans _ _ _ Hns), (ko_trans _ _ _ Hns2), (ko_trans _ _ _ Hns3),
    (ko_trans _ _ _ (ko_note_simples (ap_args ap) a3)).
  match goal with
  | |- context [add_facts ?L ?b] =>
    apply (ko_trans _ (add_facts L b)); [ apply ko_add_facts; bad_free | ]
  end.
  apply ko_cond_alias.
Qed.

Lemma ko_apply_deps (returns : list node) (exn called : node) :
  ∀ (l : list (list simple * node)) (a : acc),
    ko_ok a (unknown_arity_apply_deps returns exn called l a).
Proof.
  induction l as [ | [first w] rest IH]; intros a; [ apply ko_refl | ].
  cbn [unknown_arity_apply_deps].
  match goal with
  | |- context [add_facts ?L a] =>
    apply (ko_trans _ (add_facts L a)); [ apply ko_add_facts; bad_free | ]
  end.
  destruct rest as [ | [? next] rest'].
  - apply ko_add_facts. bad_free.
  - match goal with
    | |- context [fresh ?b] => destruct (fresh b) as [v a1] eqn:E
    end.
    ko_let E ko_fresh. apply (ko_trans _ _ _ Hns).
    match goal with
    | |- context [add_facts ?L a1] =>
      apply (ko_trans _ (add_facts L a1)); [ apply ko_add_facts; bad_free | ]
    end.
    apply IH.
Qed.

Lemma ko_make_unknown (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  ko_ok a (snd (make_unknown_arity_apply_widget env ap returns exn a)).
Proof.
  unfold make_unknown_arity_apply_widget.
  destruct (fresh a) as [called a1] eqn:E1. ko_let E1 ko_fresh.
  apply (ko_trans _ _ _ Hns).
  apply (ko_trans _ (add_fact (Fact_any_usage called) a1));
    [ apply ko_add_fact; cbn; auto | ].
  match goal with
  | |- context [fresh_list ?n ?b] => destruct (fresh_list n b) as [ws a2] eqn:E2
  end.
  ko_let E2 ko_fresh_list. apply (ko_trans _ _ _ Hns0).
  apply (ko_trans _ _ _ (ko_note_simples (ap_args ap) a2)).
  match goal with
  | |- context [unknown_arity_apply_deps ?r ?x ?c ?l ?b] =>
    apply (ko_trans _ (unknown_arity_apply_deps r x c l b));
      [ apply ko_apply_deps | ];
    destruct (fresh (unknown_arity_apply_deps r x c l b)) as [an a3] eqn:E3
  end.
  ko_let E3 ko_fresh. apply (ko_trans _ _ _ Hns1).
  destruct ws; [ apply ko_refl | apply ko_cond_alias ].
Qed.

Lemma ko_add_apply (env : traverse_env) (id : code_id) (callee widget : node)
    (only : bool) (a : acc) :
  ko_ok a (add_apply env id callee widget only a).
Proof.
  unfold add_apply. destruct only; [ | apply ko_push_apply_dep ].
  destruct (fresh a) as [c1 a1] eqn:E1. ko_let E1 ko_fresh.
  destruct (fresh a1) as [w1 a2] eqn:E2.
  assert (Hns2 : ko_ok a1 a2)
    by (change a2 with (snd (w1, a2)); rewrite <- E2; apply ko_fresh).
  apply (ko_trans _ _ _ Hns), (ko_trans _ _ _ Hns2).
  match goal with
  | |- context [add_facts ?L a2] =>
    apply (ko_trans _ (add_facts L a2)); [ apply ko_add_facts; bad_free | ]
  end.
  apply ko_push_apply_dep.
Qed.

Lemma ko_traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  ko_ok a (traverse_apply env ap a).
Proof.
  unfold traverse_apply.
  destruct (find_cont env (ap_continuation ap) _ a) as [ra a1] eqn:E1.
  ko_let E1 ko_find_cont.
  destruct (find_cont env (ap_exn_continuation ap) _ a1) as [ep a2] eqn:E2.
  assert (Hns2 : ko_ok a1 a2)
    by (change a2 with (snd (ep, a2)); rewrite <- E2; apply ko_find_cont).
  apply (ko_trans _ _ _ Hns), (ko_trans _ _ _ Hns2).
  destruct (match ep with
            | [] => fresh a2
            | exn_param :: extra_params =>
              (exn_param,
               add_facts
                 (map (fun '(p, s) => Fact_alias p (node_of_simple s))
                    (combine extra_params (exn_extra_simples ap)))
                 (note_simples (exn_extra_simples ap) a2))
            end)
    as [exn_arg a3] eqn:E3.
  assert (Hns3 : ko_ok a2 a3).
  { destruct ep as [ | p ps].
    - change a3 with (snd (exn_arg, a3)). rewrite <- E3. apply ko_fresh.
    - injection E3 as _ <-.
      apply (ko_trans _ _ _ (ko_note_simples (exn_extra_simples ap) a2)).
      apply ko_add_facts. bad_free. }
  apply (ko_trans _ _ _ Hns3).
  unfold traverse_call_kind.
  apply (ko_trans _ _ _ (ko_note_simple (ap_callee ap) a3)).
  destruct (ap_call_kind ap) as [id | | code_ids].
  - destruct (make_known_arity_apply_widget env ap ra exn_arg
                (note_simple (ap_callee ap) a3))
      as [cw a4] eqn:E4.
    assert (Hns4 : ko_ok (note_simple (ap_callee ap) a3) a4)
      by (change a4 with (snd (cw, a4)); rewrite <- E4; apply ko_make_known).
    apply (ko_trans _ _ _ Hns4).
    match goal with
    | |- context [add_fact ?f a4] =>
      apply (ko_trans _ (add_fact f a4)); [ apply ko_add_fact; cbn; auto | ]
    end.
    destruct (env_should_preserve_direct_calls env);
      try apply ko_add_apply.
    destruct (negb _); [ apply ko_add_apply | apply ko_refl ].
  - destruct (make_unknown_arity_apply_widget env ap ra exn_arg
                (note_simple (ap_callee ap) a3))
      as [cw a4] eqn:E4.
    assert (Hns4 : ko_ok (note_simple (ap_callee ap) a3) a4)
      by (change a4 with (snd (cw, a4)); rewrite <- E4;
          apply ko_make_unknown).
    apply (ko_trans _ _ _ Hns4). apply ko_add_fact. cbn. auto.
  - destruct (make_known_arity_apply_widget env ap ra exn_arg
                (note_simple (ap_callee ap) a3))
      as [cw a4] eqn:E4.
    assert (Hns4 : ko_ok (note_simple (ap_callee ap) a3) a4)
      by (change a4 with (snd (cw, a4)); rewrite <- E4; apply ko_make_known).
    apply (ko_trans _ _ _ Hns4). apply ko_add_fact. cbn. auto.
Qed.

Lemma ko_apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc) :
  ko_ok a (apply_cont_deps env ac a).
Proof.
  unfold apply_cont_deps.
  destruct (find_cont env (ac_continuation ac) _ a) as [ps a1] eqn:E1.
  ko_let E1 ko_find_cont. apply (ko_trans _ _ _ Hns).
  apply (ko_trans _ _ _ (ko_note_simples (ac_args ac) a1)).
  apply ko_add_facts. bad_free.
Qed.

Lemma ko_traverse_switch (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  ko_ok a (traverse_switch env scrutinee arms a).
Proof.
  unfold traverse_switch.
  apply (ko_trans _ _ _ (ko_cond_any_usage env scrutinee a)).
  apply ko_fold. intros a' [i ac] _. apply ko_apply_cont_deps.
Qed.

Lemma ko_exn_handler_deps (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  ko_ok a (exn_handler_deps env h a).
Proof.
  unfold exn_handler_deps. destruct h as [[ | p ps] md body];
    [ apply ko_refl | ].
  destruct (hm_is_exn_handler md); [ | apply ko_refl ].
  eapply ko_trans; [ | apply ko_cond_any_usage ].
  apply ko_add_fact. cbn. auto.
Qed.

Lemma ko_record_set (env : traverse_env)
    (closures : list (node * option code_id))
    (fs : list (function_slot * node)) (vs : list (value_slot * simple))
    (a : acc) :
  ko_ok a (record_set_of_closures_deps env closures fs vs a).
Proof.
  destruct (CE.record_set_facts env closures fs vs a) as [(L & HL & HLf) Hn].
  split; [ | split ].
  - intros fct Hin. rewrite HL. apply in_or_app. right. exact Hin.
  - intros fct Hin Hb. rewrite HL in Hin.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact Hin ].
    apply HLf in Hin
      as (base & code & _ & [(w & s & _ & ->) | (f & c & _ & ->)]);
      cbn in Hb; destruct Hb.
  - rewrite Hn. lia.
Qed.

Lemma ko_traverse_static_binding (env : traverse_env) (b : static_binding)
    (a : acc) :
  ko_ok a (traverse_static_binding env b a).
Proof.
  destruct b as [id c | id | s const];
    cbn [traverse_static_binding]; try apply ko_refl.
  destruct const as [id vs | slot cs decls vs | tag fields | contents
                    | bytes | [ | el els]].
  - apply ko_record_set.
  - apply ko_record_set.
  - apply ko_block_deps.
  - apply ko_add_facts. unfold boxed_number_facts.
    destruct (boxed_number_kind contents); [ bad_free | ].
    destruct contents; bad_free.
  - apply ko_add_fact. cbn. auto.
  - apply ko_add_fact. cbn. auto.
  - apply ko_block_deps.
Qed.

Lemma ko_traverse_named (env : traverse_env) (x : variable) (d : named)
    (a : acc) :
  ko_ok a (traverse_named env x d a).
Proof.
  destruct d as [s | p args | id vs | slot decls vs | w c | pf w c | mf mt c];
    cbn [traverse_named]; cbv zeta.
  - apply (ko_trans _ _ _ (ko_note_simple s a)). apply ko_add_fact. cbn. auto.
  - apply ko_traverse_prim.
  - apply ko_record_set.
  - unfold traverse_closure_of_set.
    destruct (among slot (map fst (compact decls))); [ | apply ko_refl ].
    destruct (closures_of_set x slot (set_closures_of env x) (compact decls) a)
      as [cl a1] eqn:Ecl.
    destruct (CE.closures_of_set_spec x slot _ _ a cl a1 Ecl)
      as (Hf1 & Hn1 & _).
    apply (ko_trans _ a1); [ apply ko_same; assumption | ].
    apply ko_record_set.
  - apply (ko_trans _ _ _ (ko_note_simple c a)). apply ko_add_fact. cbn. auto.
  - apply (ko_trans _ _ _ (ko_note_simple c a)). apply ko_add_fact. cbn. auto.
  - apply (ko_trans _ _ _ (ko_note_simple c a)). apply ko_add_fact. cbn. auto.
Qed.

Lemma ko_code_aliases (my_closure : variable) (params : list variable)
    (cd : code_dep) (a : acc) :
  ko_ok a
    (add_facts
       (Fact_alias (Node_variable my_closure) (cd_my_closure cd)
        :: map (fun '(p, arg) => Fact_alias (Node_variable p) arg)
             (combine params (cd_params cd)))
       a).
Proof. apply ko_add_facts. bad_free. Qed.

Lemma ko_add_closure_dep (code_deps : finite_map code_dep) (d : closure_dep)
    (a : acc) :
  ko_ok a (add_closure_dep code_deps d a).
Proof.
  unfold add_closure_dep.
  destruct (lookup (cld_closure_code_id d) code_deps) as [cd | ].
  - apply ko_add_facts.
    destruct (cd_unknown_arity_call_witnesses cd); bad_free.
  - destruct (fresh a) as [w a1] eqn:E. ko_let E ko_fresh.
    apply (ko_trans _ _ _ Hns). apply ko_add_facts. bad_free.
Qed.

Lemma ko_resolve (unit_symbols : list symbol) (a : acc) :
  ko_ok a (resolve_delayed_deps unit_symbols a).
Proof.
  unfold resolve_delayed_deps. cbv zeta.
  match goal with
  | |- context [add_facts (flat_map ?f ?l) a] =>
    apply (ko_trans _ (add_facts (flat_map f l) a))
  end.
  { apply ko_add_facts. intros fct Hin Hs.
    apply in_flat_map in Hin as (d & _ & Hin). unfold apply_dep_facts in Hin.
    destruct (lookup _ _); [ | destruct (ad_function_containing_apply_expr d) ];
      revert fct Hin Hs; try (destruct (ad_function_containing_apply_expr d));
      bad_free. }
  cbv zeta.
  match goal with
  | |- context [List.fold_left ?f ?l ?b] =>
    apply (ko_trans _ (List.fold_left f l b));
      [ apply ko_fold; intros; apply ko_add_closure_dep | ]
  end.
  apply ko_add_facts. bad_free.
Qed.


Lemma ko_traverse :
  (∀ (e : expr) (env : traverse_env) (a : acc), ko_ok a (traverse env e a))
  ∧ ∀ h : continuation_handler,
      match h with
      | Handler _ _ handler =>
        ∀ (env : traverse_env) (a : acc), ko_ok a (traverse env handler a)
      end.
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x k d body IH env a. cbn [traverse].
    exact (ko_trans _ _ _ (ko_traverse_named env x d a) (IH _ _)).
  - intros bindings body Hcodes IH env a. cbn [traverse].
    eapply ko_trans; [ | apply IH ].
    match goal with
    | |- ko_ok a (List.fold_left _ bindings ?b) =>
      apply (ko_trans _ b); [ | apply ko_fold ]
    end.
    2:{ intros a' [id c | id | s const] Hb;
        cbn [traverse_binding]; [ | apply ko_refl | apply ko_refl ].
        destruct c as [params mc ret exn md cbody]. cbn [traverse_code].
        destruct (lookup id (acc_code_deps a')) as [cd | ];
          [ | apply ko_refl ].
        pose proof (Hcodes id (Code params mc ret exn md cbody) Hb) as Hc.
        cbn in Hc. eapply ko_trans; [ apply ko_code_aliases | apply Hc ]. }
    match goal with
    | |- ko_ok a (List.fold_left _ bindings ?b) =>
      apply (ko_trans _ b);
        [ | apply ko_fold; intros; apply ko_traverse_static_binding ]
    end.
    apply ko_fold. intros. apply ko_prepare_binding.
  - intros k h body IHh IH env a. destruct h as [params md handler].
    cbn [traverse traverse_handler] in IHh |- *.
    eapply ko_trans; [ | apply IH ].
    eapply ko_trans; [ apply IHh | apply ko_exn_handler_deps ].
  - intros handlers body IHh IH env a. cbn [traverse].
    eapply ko_trans; [ | apply IH ].
    apply ko_fold. intros a' [k h] Hkh. destruct h as [params md handler].
    pose proof (IHh k _ Hkh) as Hh. cbn [traverse_handler] in Hh |- *.
    apply Hh.
  - intros ap env a. apply ko_traverse_apply.
  - intros ac env a. apply ko_apply_cont_deps.
  - intros scrutinee arms env a. apply ko_traverse_switch.
  - intros env a. apply ko_refl.
  - intros params md handler IH. exact IH.
Qed.

(* The traversal of a unit makes no bad fact. *)
Lemma unit_no_bad (flag : reaper_preserve_direct_calls) (u : unit_interface)
    (e : expr) :
  ∃ a : acc,
    traverse_unit flag u e = facts_of_list (acc_facts a)
    ∧ ∀ fct : fact, fct ∈ acc_facts a -> ~ bad_fact fct.
Proof.
  unfold traverse_unit.
  set (a0 := add_fact (Fact_any_source Node_all_constants) empty_acc).
  destruct (fresh a0) as [r a1] eqn:E1.
  destruct (fresh a1) as [x a2] eqn:E2.
  set (a3 := add_fact (Fact_any_usage x) a2).
  match goal with
  | |- context [traverse ?env e a3] => set (env0 := env)
  end.
  eexists. split; [ reflexivity | ].
  assert (Hf3 : ∀ fct, fct ∈ acc_facts a3 -> ~ bad_fact fct).
  { assert (Hf : acc_facts a3
                 = [Fact_any_usage x; Fact_any_source Node_all_constants]).
    { unfold a3. cbn [add_fact add_facts acc_facts app].
      assert (Hf2 : acc_facts a2 = acc_facts a1)
        by (change a2 with (snd (x, a2)); rewrite <- E2; reflexivity).
      assert (Hf1 : acc_facts a1 = acc_facts a0)
        by (change a1 with (snd (r, a1)); rewrite <- E1; reflexivity).
      rewrite Hf2, Hf1. reflexivity. }
    rewrite Hf. intros fct [<- | [<- | []]]; cbn; auto. }
  assert (Hko : ko_ok a3
                  (resolve_delayed_deps (static_names symbols_of_binding e)
                     (add_fact (Fact_any_usage r) (traverse env0 e a3)))).
  { eapply ko_trans; [ | apply ko_resolve ].
    eapply ko_trans; [ apply (proj1 ko_traverse) | ].
    apply ko_add_fact. cbn. auto. }
  intros fct Hin Hb. exact (Hf3 fct (proj1 (proj2 Hko) fct Hin Hb) Hb).
Qed.

End Kinds.

Theorem block_fields_of_one_kind_traversal :
  block_fields_of_one_kind_traversal_statement.
Proof.
  intros flag u e a from from' i k k' H H'.
  destruct (Kinds.unit_no_bad flag u e) as (a0 & Ha & Hok).
  assert (Hv : ∀ (kk : flambda_kind) (fr : node),
             g_constructor (graph_of_facts (traverse_unit flag u e)) a
               (Field_block i kk) fr -> kk = Value).
  { intros kk fr Hc. rewrite Ha in Hc.
    apply CE.constructor_of_list in Hc.
    destruct kk; [ reflexivity | | | ];
      exfalso; exact (Hok _ Hc I). }
  rewrite (Hv k from H), (Hv k' from' H'). reflexivity.
Qed.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Theorem rewrite_changed_type_sound_traversal :
  rewrite_changed_type_sound_traversal_statement.
Proof.
  intros is_local u T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' fuel m t n v v_mid v' Hfrag Hwfu Honce
    Hleast Hlocal Hsets Hkinds Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc
    Huses Hunst Hone Ht Hcis Hrv Hat Hch Hsat.
  destruct (traversal_props u e Hfrag Hwfu Honce) as [Hmut Hfs].
  apply (TP.rewrite_changed_type_sound_slots is_local (traverse_unit Auto u e)
           T sets dominated to_unbox tcr e tenv canon statics statics'
           valuation valuation' fuel m t n v v_mid v');
    try assumption.
  exact (reprs_of_sets_derived is_local _ T dominated to_unbox tcr e Hleast
           Hfs Hmut Hdomc Hone).
Qed.

Theorem rewrite_changed_type_sound_traversal_landed :
  rewrite_changed_type_sound_traversal_landed_statement.
Proof.
  intros is_local u T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' fuel m t n v v_mid v' Hfrag Hwfu Honce
    Hleast Hlocal Hsets Hkinds Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc
    Huses Hunst Hone Ht Hcis Hrv Hat Hch Hsat.
  destruct (traversal_props u e Hfrag Hwfu Honce) as [Hmut Hfs].
  apply (TP.rewrite_changed_type_sound_slots_landed is_local
           (traverse_unit Auto u e) T sets dominated to_unbox tcr e tenv
           canon statics statics' valuation valuation' fuel m t n v v_mid v');
    try assumption.
  exact (reprs_of_sets_derived is_local _ T dominated to_unbox tcr e Hleast
           Hfs Hmut Hdomc Hone).
Qed.

Theorem changed_type_sound_traversal : changed_type_sound_traversal_statement.
Proof.
  intros is_local u T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' fuel m t n v v_mid v' Hfrag Hwfu Honce
    Hleast Hlocal Hsets Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc Huses
    Hunst Hu Ht Hcis Hrv Hat Hch Hsat.
  destruct (traversal_props u e Hfrag Hwfu Honce) as [Hmut Hfs].
  exact (rewrite_changed_type_sound_traversal is_local u T sets dominated
           to_unbox tcr e tenv canon statics statics' valuation valuation'
           fuel m t n v v_mid v' Hfrag Hwfu Honce Hleast Hlocal Hsets
           (block_fields_of_one_kind_traversal Auto u e) Henv Hfragt Halloc
           Hst Hwhole Hcopies Hdomc Huses Hunst
           (changed_closures_one_allocation_unaliased is_local _ T tcr Hleast
              Hfs Hmut Hu)
           Ht Hcis Hrv Hat Hch Hsat).
Qed.

Theorem changed_type_sound_traversal_landed :
  changed_type_sound_traversal_landed_statement.
Proof.
  intros is_local u T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' fuel m t n v v_mid v' Hfrag Hwfu Honce
    Hleast Hlocal Hsets Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc Huses
    Hunst Hu Ht Hcis Hrv Hat Hch Hsat.
  destruct (traversal_props u e Hfrag Hwfu Honce) as [Hmut Hfs].
  exact (rewrite_changed_type_sound_traversal_landed is_local u T sets dominated
           to_unbox tcr e tenv canon statics statics' valuation valuation'
           fuel m t n v v_mid v' Hfrag Hwfu Honce Hleast Hlocal Hsets
           (block_fields_of_one_kind_traversal Auto u e) Henv Hfragt Halloc
           Hst Hwhole Hcopies Hdomc Huses Hunst
           (changed_closures_one_allocation_unaliased is_local _ T tcr Hleast
              Hfs Hmut Hu)
           Ht Hcis Hrv Hat Hch Hsat).
Qed.

(* ======================================================================== *)
(* A changed closure stored in a block                                      *)
(* ======================================================================== *)

Theorem changed_unstored_counterexample :
  changed_unstored_counterexample_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100).
    vm_compute. reflexivity. }
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  intros H.
  assert (Hc : g_constructor (graph_of_facts StoredExample.facts)
                 (Node_variable 10%positive) (Field_block 0 Value)
                 (Node_variable 60%positive)).
  { cbv [graph_of_facts g_constructor].
    match goal with
    | |- In ?x ?l =>
      let l' := eval vm_compute in l in
      change (In x l'); repeat first [ left; reflexivity | right ]
    end. }
  assert (Hmay : may_come_from (solution_of_tables StoredExample.tables)
                   (Node_variable 60%positive) (Node_variable 60%positive)).
  { split; [ vm_compute; reflexivity | ].
    split; [ right; vm_compute; reflexivity | ].
    split; [ intros E; exact E | split; [ intros E; exact E | ] ].
    intros _ u Hu. left. exact Hu. }
  pose proof (H _ _ _ _ Hc (fun g E => ltac:(discriminate E)) Hmay
                ltac:(vm_compute; reflexivity)) as E.
  vm_compute in E. discriminate E.
Qed.
