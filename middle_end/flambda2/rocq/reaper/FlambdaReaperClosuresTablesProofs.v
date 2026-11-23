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

(* The proofs of FlambdaReaperClosuresTablesSpec.v. Nothing here needs
   review.

   [binv_fold]: each representation of the fold of
   [changed_representations] has new function slots that are distinct
   and from its start on, and new value slots that are distinct
   ([mk_unboxed_fields_fresh]). The start is after every slot of the
   unit, so that renaming is injective on them ([rename_inj]).

   [changed_slot_current]: the traversal holds the variable of a [Let]
   of a closure at its slot in its own allocation ([let_edges], from
   [traverse_unit_facts]); [rinv] of
   FlambdaReaperTypesTraversalProofs.v gives the closure of its
   representation in the set of a node of [tcr], which holds it at
   [cr_current r] and so at the slot too ([function_slots_of_sets]),
   one slot only.

   [repr_unboxing_frag_sound]: by induction on the unit, every [Let] of
   it being among its [let_bindings] ([frag_mono]). *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaNameAbstractionSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesClosuresSpec FlambdaReaperClosuresTablesSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs FlambdaReaperTypesProofs
  FlambdaReaperTypesTraversalSpec FlambdaReaperTypesTraversalProofs
  FlambdaReaperTypesTraversalUnaliasedProofs
  FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesClosuresProofs FlambdaNameAbstractionProofs.
From OxCamlFlambda2 Require FlambdaReaperUnboxingFuelProofs.
Import ListNotations.

Module UFP := FlambdaReaperUnboxingFuelProofs.

Module TP := FlambdaReaperTypesProofs.
Module TTP := FlambdaReaperTypesTraversalProofs.
Module URP := FlambdaReaperUnboxingRefinesProofs.
Module FP := FlambdaReaperFlowProofs.
Module TUP := FlambdaReaperTypesTraversalUnaliasedProofs.

(* ======================================================================== *)
(* The representations, for any tables                                      *)
(* ======================================================================== *)

Lemma rename_next (L : list function_slot) :
  ∀ n : positive, (n <= snd (rename_slots L n))%positive.
Proof.
  induction L as [ | fs L IH]; intros n; cbn [rename_slots];
    [ cbn [snd]; lia | ].
  specialize (IH (Pos.succ n)).
  destruct (rename_slots L (Pos.succ n)) as [rest n'] eqn:E.
  cbn [snd] in *. lia.
Qed.

(* What every representation of the fold has, from the start [n0]. *)
Definition binv (n0 : positive) (acc : list (node * closure_repr)) : Prop :=
  ∀ (c : node) (r : closure_repr),
    (c, r) ∈ acc ->
    NoDup (map snd (cr_function_slots r))
    ∧ (∀ q : positive,
         q ∈ map snd (cr_function_slots r) -> (n0 <= q)%positive)
    ∧ NoDup (URP.leaf_variables (cr_fields r)).

Lemma binv_fold (F : facts) (T : tables) (cm : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node)
    (is_local : field -> bool) (tcr : list node) (n0 : positive) :
  binv n0
    (fst (changed_representations F T cm dominated to_unbox is_local tcr n0)).
Proof.
  rewrite UFP.changed_representations_depth.
  match goal with
  | |- context [List.fold_left ?h tcr _] => set (step := h)
  end.
  assert (Hgen :
    ∀ (l : list node) (acc : list (node * closure_repr)) (seen : list node)
      (n : positive),
      (n0 <= n)%positive -> binv n0 acc ->
      (n0 <= snd (List.fold_left step l (acc, seen, n)))%positive
      ∧ binv n0 (fst (fst (List.fold_left step l (acc, seen, n))))).
  { induction l as [ | x l IH]; intros acc seen n Hn Hinv;
      [ split; [ exact Hn | exact Hinv ] | ].
    cbn [List.fold_left].
    destruct (step (acc, seen, n) x) as [[acc' seen'] n'] eqn:Hstep.
    apply IH.
    - unfold step in Hstep. cbv beta match in Hstep.
      destruct (mem_node x seen);
        [ injection Hstep as <- <- <-; exact Hn | ].
      destruct (set_function_slots F x) as [ | p0 l0];
        [ injection Hstep as <- <- <-; exact Hn | ].
      match type of Hstep with
      | context [mk_unboxed_fields ?F' ?T' ?cm' ?d' ?tu' ?fuel ?b ?fl n] =>
        pose proof (URP.mk_unboxed_fields_fresh F' T' cm' d' tu' fuel b fl n)
          as Hfr;
        cbv zeta in Hfr;
        destruct (mk_unboxed_fields F' T' cm' d' tu' fuel b fl n)
          as [fields n1] eqn:Hmk
      end.
      destruct (rename_slots (map fst (p0 :: l0)) n1) as [fss n2] eqn:Hren.
      injection Hstep as <- <- <-.
      pose proof (rename_next (map fst (p0 :: l0)) n1) as Hn2.
      rewrite Hren in Hn2. cbn [snd] in Hn2.
      destruct Hfr as (Hn1 & _). cbn [snd] in Hn1. lia.
    - unfold step in Hstep. cbv beta match in Hstep.
      destruct (mem_node x seen);
        [ injection Hstep as <- <- <-; exact Hinv | ].
      destruct (set_function_slots F x) as [ | p0 l0] eqn:Hsx;
        [ injection Hstep as <- <- <-; exact Hinv | ].
      match type of Hstep with
      | context [mk_unboxed_fields ?F' ?T' ?cm' ?d' ?tu' ?fuel ?b ?fl n] =>
        pose proof (URP.mk_unboxed_fields_fresh F' T' cm' d' tu' fuel b fl n)
          as Hfr;
        cbv zeta in Hfr;
        destruct (mk_unboxed_fields F' T' cm' d' tu' fuel b fl n)
          as [fields n1] eqn:Hmk
      end.
      destruct (rename_slots (map fst (p0 :: l0)) n1) as [fss n2] eqn:Hren.
      injection Hstep as <- <- <-.
      destruct Hfr as (Hn1 & Hfnd & _). cbn [fst snd] in Hn1, Hfnd.
      pose proof (TTP.rename_nodup (map fst (p0 :: l0)) n1) as [Hnd Hge].
      rewrite Hren in Hnd, Hge. cbn [fst] in Hnd, Hge.
      intros c r Hin. apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hinv c r Hin) | ].
      assert (Hin' :
        (c, r)
        ∈ flat_map
            (fun '(fs, f) =>
               flat_map
                 (fun '(a, c) =>
                    if node_eqb a f
                    then [(c, Closure_repr fields fss fs)]
                    else [])
                 dominated)
            (p0 :: l0)) by exact Hin.
      apply TTP.in_new in Hin' as (fs & f & _ & _ & ->).
      cbn [cr_function_slots cr_fields].
      split; [ exact Hnd | split; [ | exact Hfnd ] ].
      intros q Hq. pose proof (Hge q Hq). lia. }
  destruct (List.fold_left step tcr ([], [], n0)) as [[acc seen] n] eqn:Hf.
  cbn [fst].
  pose proof (Hgen tcr [] [] n0 (Pos.le_refl n0)
                (fun c r H => match H with end)) as [_ H].
  rewrite Hf in H. exact H.
Qed.

Lemma changed_repr_props (is_local : field -> bool) (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox tcr : list node) (e : expr)
    (n : node) (r : closure_repr) :
  Keyed.lookup node_eqb n
    (unit_changed_representations is_local F T dominated to_unbox tcr e)
  = Some r ->
  NoDup (map snd (cr_function_slots r))
  ∧ (∀ q : positive,
       q ∈ map snd (cr_function_slots r) ->
       (fresh_name (slots_of e) <= q)%positive)
  ∧ NoDup (URP.leaf_variables (cr_fields r)).
Proof.
  intros Hlk. apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hlk.
  exact (binv_fold F T (bound_code_my_closures e) dominated to_unbox is_local
           tcr (fresh_name (slots_of e)) n r Hlk).
Qed.

Theorem changed_leaves_distinct : changed_leaves_distinct_statement.
Proof.
  intros is_local F T dominated to_unbox tcr e n r Hlk.
  destruct (changed_repr_props is_local F T dominated to_unbox tcr e n r Hlk)
    as (_ & _ & H).
  apply distinct_NoDup. exact H.
Qed.

(* ======================================================================== *)
(* Renaming the function slots                                              *)
(* ======================================================================== *)

Lemma lookup_inj (l : list (positive * positive)) (p q v : positive) :
  NoDup (map snd l) -> lookup p l = Some v -> lookup q l = Some v -> p = q.
Proof.
  induction l as [ | [k w] l IH]; intros Hnd Hp Hq; [ discriminate Hp | ].
  rewrite lookup_cons in Hp, Hq. cbn [map snd] in Hnd.
  apply NoDup_cons_iff in Hnd as [Hw Hnd].
  destruct (Pos.eqb_spec p k) as [-> | Hpk];
    destruct (Pos.eqb_spec q k) as [-> | Hqk].
  - reflexivity.
  - injection Hp as <-. exfalso. apply Hw.
    exact (in_map snd l (q, w) (lookup_in _ _ _ Hq)).
  - injection Hq as <-. exfalso. apply Hw.
    exact (in_map snd l (p, w) (lookup_in _ _ _ Hp)).
  - exact (IH Hnd Hp Hq).
Qed.

Lemma rename_inj (r : closure_repr) (N p q : positive) :
  NoDup (map snd (cr_function_slots r)) ->
  (∀ s : positive, s ∈ map snd (cr_function_slots r) -> (N <= s)%positive) ->
  (p < N)%positive -> (q < N)%positive ->
  rename_function_slot r p = rename_function_slot r q -> p = q.
Proof.
  intros Hnd Hge Hp Hq. unfold rename_function_slot.
  destruct (lookup p (cr_function_slots r)) as [p' | ] eqn:Ep;
    destruct (lookup q (cr_function_slots r)) as [q' | ] eqn:Eq; intros E.
  - subst q'. exact (lookup_inj _ p q p' Hnd Ep Eq).
  - subst q.
    pose proof (Hge p' (in_map snd _ (p, p') (lookup_in _ _ _ Ep))). lia.
  - subst p.
    pose proof (Hge q' (in_map snd _ (q, q') (lookup_in _ _ _ Eq))). lia.
  - exact E.
Qed.

Lemma rename_decls_cons (r : closure_repr) (fs : function_slot)
    (id : code_id) (decls : list (function_slot * code_id)) :
  rename_function_decls r ((fs, id) :: decls)
  = (rename_function_slot r fs, id) :: rename_function_decls r decls.
Proof. reflexivity. Qed.

Lemma lookup_renamed (r : closure_repr) (N slot : positive)
    (decls : list (function_slot * code_id)) :
  NoDup (map snd (cr_function_slots r)) ->
  (∀ s : positive, s ∈ map snd (cr_function_slots r) -> (N <= s)%positive) ->
  (slot < N)%positive ->
  (∀ fs : function_slot, fs ∈ map fst decls -> (fs < N)%positive) ->
  lookup (rename_function_slot r slot) (rename_function_decls r decls)
  = lookup slot decls.
Proof.
  intros Hnd Hge Hs Hd.
  induction decls as [ | [fs id] decls IH]; [ reflexivity | ].
  rewrite rename_decls_cons, !lookup_cons.
  assert (Hfs : (fs < N)%positive) by (apply Hd; left; reflexivity).
  destruct (Pos.eqb_spec slot fs) as [-> | Hne].
  - rewrite Pos.eqb_refl. reflexivity.
  - destruct (Pos.eqb_spec (rename_function_slot r slot)
                (rename_function_slot r fs)) as [E | _].
    + exfalso. exact (Hne (rename_inj r N slot fs Hnd Hge Hs Hfs E)).
    + apply IH. intros fs' H. apply Hd. right. exact H.
Qed.

(* ======================================================================== *)
(* The [Let]s of a unit                                                     *)
(* ======================================================================== *)

Lemma lt_fresh_name (names : list positive) (p : positive) :
  p ∈ names -> (p < fresh_name names)%positive.
Proof.
  intros H. pose proof (FlambdaNameAbstractionProofs.fresh_name_above names p H)
    as Hle.
  unfold fresh_name. lia.
Qed.

(* The slots of the binding of a [Let] of a term are slots of the term,
   and the binding is well formed if the sets of the term are. *)
Definition let_ok_P (e : expr) : Prop :=
  ∀ (x : variable) (d : named),
    (x, d) ∈ let_bindings e ->
    (∀ p : positive, p ∈ slots_of_named d -> p ∈ slots_of e)
    ∧ (sets_well_formed e = true -> named_sets_well_formed d = true).

Lemma let_ok :
  (∀ e : expr, let_ok_P e)
  ∧ ∀ h : continuation_handler,
      match h with Handler _ _ handler => let_ok_P handler end.
Proof.
  apply FP.expr_handler_ind; unfold let_ok_P.
  - intros x0 kind d0 body IH x d Hin.
    cbn [let_bindings slots_of sets_well_formed] in *.
    destruct Hin as [[= <- <-] | Hin].
    + split; [ intros p Hp; apply in_or_app; left; exact Hp | ].
      intros Hwf. apply andb_true_iff in Hwf as [Hwf _]. exact Hwf.
    + destruct (IH x d Hin) as [H1 H2].
      split; [ intros p Hp; apply in_or_app; right; exact (H1 p Hp) | ].
      intros Hwf. apply andb_true_iff in Hwf as [_ Hwf]. exact (H2 Hwf).
  - intros bindings body Hcodes IH x d Hin.
    cbn [let_bindings slots_of sets_well_formed] in *.
    apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as (b & Hb & Hin).
      destruct b as [id c | id | s const]; cbn in Hin; [ | destruct Hin
                                                       | destruct Hin ].
      destruct c as [params mc ret exn md cbody]. cbn in Hin.
      destruct (Hcodes id _ Hb x d Hin) as [H1 H2].
      split.
      * intros p Hp. apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id (Code params mc ret exn md cbody)).
        split; [ exact Hb | exact (H1 p Hp) ].
      * intros Hwf. apply andb_true_iff in Hwf as [Hwf _].
        apply andb_true_iff in Hwf as [_ Hwf].
        rewrite forallb_forall in Hwf. apply H2. exact (Hwf _ Hb).
    + destruct (IH x d Hin) as [H1 H2].
      split; [ intros p Hp; apply in_or_app; right; exact (H1 p Hp) | ].
      intros Hwf. apply andb_true_iff in Hwf as [_ Hwf]. exact (H2 Hwf).
  - intros k h body IHh IH x d Hin. destruct h as [params md handler].
    cbn [let_bindings handler_let_bindings slots_of handler_slots
         sets_well_formed handler_sets_well_formed] in *.
    apply in_app_or in Hin as [Hin | Hin].
    + destruct (IHh x d Hin) as [H1 H2].
      split; [ intros p Hp; apply in_or_app; left; exact (H1 p Hp) | ].
      intros Hwf. apply andb_true_iff in Hwf as [Hwf _]. exact (H2 Hwf).
    + destruct (IH x d Hin) as [H1 H2].
      split; [ intros p Hp; apply in_or_app; right; exact (H1 p Hp) | ].
      intros Hwf. apply andb_true_iff in Hwf as [_ Hwf]. exact (H2 Hwf).
  - intros handlers body IHh IH x d Hin.
    cbn [let_bindings slots_of sets_well_formed] in *.
    apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as ([k h] & Hkh & Hin).
      destruct h as [params md handler].
      cbn [handler_let_bindings] in Hin.
      destruct (IHh k _ Hkh x d Hin) as [H1 H2].
      split.
      * intros p Hp. apply in_or_app. left. apply in_flat_map.
        exists (k, Handler params md handler).
        split; [ exact Hkh | exact (H1 p Hp) ].
      * intros Hwf. apply andb_true_iff in Hwf as [Hwf _].
        rewrite forallb_forall in Hwf. apply H2. exact (Hwf _ Hkh).
    + destruct (IH x d Hin) as [H1 H2].
      split; [ intros p Hp; apply in_or_app; right; exact (H1 p Hp) | ].
      intros Hwf. apply andb_true_iff in Hwf as [_ Hwf]. exact (H2 Hwf).
  - intros ap x d [].
  - intros ac x d [].
  - intros scrutinee arms x d [].
  - intros x d [].
  - intros params md handler IH. exact IH.
Qed.

Theorem changed_decls_renamed : changed_decls_renamed_statement.
Proof.
  intros is_local F T dominated to_unbox tcr e x slot decls vs n r Hx Hlk.
  destruct (changed_repr_props is_local F T dominated to_unbox tcr e n r Hlk)
    as (Hnd & Hge & _).
  destruct (proj1 let_ok e x _ Hx) as [Hslots _].
  apply (lookup_renamed r (fresh_name (slots_of e))); [ exact Hnd | exact Hge
                                                     | | ].
  - apply lt_fresh_name. apply Hslots. left. reflexivity.
  - intros fs Hfs. apply lt_fresh_name. apply Hslots. right.
    apply in_or_app. left. exact Hfs.
Qed.

(* ======================================================================== *)
(* The slot of the closure                                                  *)
(* ======================================================================== *)

Section Edges.

Variable G : graph.
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Variable cds : finite_map code_dep.
Variable keep : value_slot -> bool.
Variable codes : list (code_id * code).
Variable symbols : list symbol.
Variable slots_read : Prop.
Variable heap_prims : Prop.

Definition edges_P (e : expr) : Prop :=
  ∀ (ctx : option code_id) (ce : finite_map (list node)),
    FP.expr_edges G noted deps cds keep codes symbols slots_read heap_prims
      ctx ce e ->
    ∀ (x : variable) (d : named),
      (x, d) ∈ let_bindings e ->
      ∃ ctx' : option code_id,
        FP.named_edges G noted deps keep slots_read heap_prims ctx' x d.

(* The edges of every [Let] of a term. *)
Lemma let_edges :
  (∀ e : expr, edges_P e)
  ∧ ∀ h : continuation_handler,
      match h with Handler _ _ handler => edges_P handler end.
Proof.
  apply FP.expr_handler_ind; unfold edges_P.
  - intros x0 kind d0 body IH ctx ce [Hd Hb] x d Hin.
    cbn [let_bindings] in Hin.
    destruct Hin as [[= <- <-] | Hin];
      [ exists ctx; exact Hd | exact (IH ctx ce Hb x d Hin) ].
  - intros bindings body Hcodes IH ctx ce [Hbs Hb] x d Hin.
    cbn [let_bindings] in Hin.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH ctx ce Hb x d Hin) ].
    apply in_flat_map in Hin as (b & Hbin & Hin).
    destruct b as [id c | id | s const]; cbn in Hin; [ | destruct Hin
                                                     | destruct Hin ].
    pose proof (proj1 (every_in _ _ bindings) Hbs _ Hbin) as Hbe.
    cbn [FP.binding_edges] in Hbe. destruct Hbe as [_ Hce].
    destruct c as [params mc ret exn md cbody]. cbn in Hin.
    cbn [FP.code_edges] in Hce.
    destruct Hce as (cd & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & Hce).
    exact (Hcodes id _ Hbin _ _ Hce x d Hin).
  - intros k h body IHh IH ctx ce [Hh Hb] x d Hin.
    destruct h as [params md handler].
    cbn [let_bindings handler_let_bindings] in Hin.
    cbn [FP.handler_edges] in Hh.
    apply in_app_or in Hin as [Hin | Hin];
      [ exact (IHh _ _ Hh x d Hin) | exact (IH _ _ Hb x d Hin) ].
  - intros handlers body IHh IH ctx ce [Hhs Hb] x d Hin.
    cbn [let_bindings] in Hin.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH _ _ Hb x d Hin) ].
    apply in_flat_map in Hin as ([k h] & Hkh & Hin).
    pose proof (proj1 (every_in _ _ handlers) Hhs _ Hkh) as Hh.
    destruct h as [params md handler].
    cbn [handler_let_bindings] in Hin. cbn [FP.handler_edges] in Hh.
    exact (IHh k _ Hkh _ _ Hh x d Hin).
  - intros ap ctx ce _ x d [].
  - intros ac ctx ce _ x d [].
  - intros scrutinee arms ctx ce _ x d [].
  - intros ctx ce _ x d [].
  - intros params md handler IH. exact IH.
Qed.

End Edges.

(* The variable of a [Let] of a closure is held at its slot by itself. *)
Lemma closure_let_self (u : unit_interface) (e : expr) (x : variable)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) :
  in_flow_fragment e = true -> well_formed_unit u e = true ->
  (x, Closure_of_set slot decls vs) ∈ let_bindings e ->
  g_constructor (graph_of_facts (traverse_unit Auto u e)) (Node_variable x)
    (Field_function_slot slot) (Node_variable x).
Proof.
  intros Hflow Hwfu Hx.
  assert (Hsw : sets_well_formed e = true).
  { unfold well_formed_unit in Hwfu.
    apply andb_true_iff in Hwfu as [Hwfu _].
    apply andb_true_iff in Hwfu as [Hwfu _].
    apply andb_true_iff in Hwfu as [Hwfu _].
    apply andb_true_iff in Hwfu as [_ Hwfu]. exact Hwfu. }
  pose proof (proj2 (proj1 let_ok e x _ Hx) Hsw) as Hn.
  cbn [named_sets_well_formed set_well_formed] in Hn.
  apply andb_true_iff in Hn as [Hn _]. apply andb_true_iff in Hn as [Hn _].
  unfold is_bound in Hn.
  destruct (lookup slot decls) as [id0 | ] eqn:Hl; [ | discriminate Hn ].
  destruct (FP.traverse_unit_facts Auto u e True Hflow (or_introl I))
    as (noted & deps & cds & Hedges & _).
  destruct (proj1 (let_edges _ noted deps cds _ _ _ True False) e None
              (unit_conts u) Hedges x _ Hx)
    as (ctx & Hset).
  cbn [FP.named_edges] in Hset.
  destruct (Hset id0 Hl) as (sites & Hsx & Hall).
  destruct (Hall slot id0 Hl) as (_ & Hfs & _).
  pose proof (Hfs slot id0 Hl) as H. rewrite Hsx in H. exact H.
Qed.

Theorem changed_slot_current : changed_slot_current_statement.
Proof.
  intros is_local u e T dominated to_unbox tcr x slot decls vs r Hflow Hwfu
    Honce Hleast Hdom Hx Hlk.
  set (F := traverse_unit Auto u e) in *.
  pose proof (TTP.function_slots_of_sets_traversal u e Hflow Hwfu Honce)
    as Hsets.
  pose proof (TTP.function_slots_mutual_traversal u e Hflow Hwfu Honce)
    as Hmut.
  pose proof (TUP.closures_unaliased_traversal u e Hflow Hwfu Honce) as Hu.
  pose proof (TTP.changed_closures_one_allocation_unaliased is_local F T tcr
                Hleast Hsets Hmut Hu) as Hone.
  pose proof (TP.least_closed_tables is_local F T (or_introl Hleast)) as Hc.
  pose proof (closure_let_self u e x slot decls vs Hflow Hwfu Hx) as Hxx.
  unfold unit_changed_representations in Hlk.
  destruct (TTP.rinv_fold is_local F T (bound_code_my_closures e) dominated
              to_unbox tcr Hc Hsets Hmut Hdom Hone (fresh_name (slots_of e)))
    as (seen & Hinv1 & _).
  destruct (Hinv1 (Node_variable x) r slot (Node_variable x) Hxx Hlk)
    as (x0 & _ & Hcur & _).
  pose proof (proj1 (TTP.sfs_iff F Hsets x0 (Node_variable x) (cr_current r))
                Hcur) as Hx0.
  destruct Hsets as (_ & Hinj & Hcl).
  pose proof (proj2 (Hcl x0 (Node_variable x) (Node_variable x) (cr_current r)
                       slot Hx0) Hxx) as Hx0s.
  exact (Hinj x0 (Node_variable x) slot (cr_current r) Hx0s Hx0).
Qed.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Ltac split_bools :=
  repeat match goal with
  | H : (_ && _) = true |- _ => apply andb_true_iff in H as [? ?]
  | |- (_ && _) = true => apply andb_true_iff; split
  end.

Section Mono.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).
Variable my_closures : finite_map variable.
Variable nr : code_id -> bool.
Variable L0 : list (variable * named).

(* The conjuncts that [changed_set_frag_ok] drops hold of the [Let]s
   among [L0]. *)
Hypothesis Hcs : ∀ (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (r : closure_repr),
  (x, d) ∈ L0 -> changed_name changed (Node_variable x) = Some r ->
  changed_set_frag_ok F sol unboxed changed x kind d body r = true ->
  changed_set_ok F sol unboxed changed x kind d body r = true.

Abbreviation frag kinds C e :=
  (repr_expr_frag F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr kinds C e).
Abbreviation full kinds C e :=
  (repr_expr F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr kinds C e).
Abbreviation frag_h kinds C ds h :=
  (repr_handler_frag F sol cannot_change code_deps unit_code_ids codes
     unboxed changed my_closures nr kinds C ds h).
Abbreviation full_h kinds C ds h :=
  (repr_handler F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr kinds C ds h).

Definition mono_P (e : expr) : Prop :=
  ∀ (kinds : finite_map flambda_kind) (C : finite_map (list decision * bool)),
    (∀ p, p ∈ let_bindings e -> p ∈ L0) ->
    frag kinds C e = true -> full kinds C e = true.

Lemma frag_mono :
  (∀ e : expr, mono_P e)
  ∧ ∀ h : continuation_handler,
      match h with
      | Handler params md handler =>
        ∀ (kinds : finite_map flambda_kind)
          (C : finite_map (list decision * bool)) (ds : list decision),
          (∀ p, p ∈ let_bindings handler -> p ∈ L0) ->
          frag_h kinds C ds (Handler params md handler) = true ->
          full_h kinds C ds (Handler params md handler) = true
      end.
Proof.
  apply FP.expr_handler_ind; unfold mono_P.
  - intros x kind d body IH kinds C Hsub H.
    cbn [repr_expr_frag] in H. cbn [repr_expr].
    apply andb_true_iff in H as [H Hb]. apply andb_true_iff in H as [Hk Hd].
    rewrite Hk, (IH _ C (fun p Hp => Hsub p (or_intror Hp)) Hb).
    rewrite andb_true_r. cbn [andb].
    destruct (fields_of unboxed x) as [fields | ];
      destruct (changed_name changed (Node_variable x)) as [r | ] eqn:Hch;
      try exact Hd.
    exact (Hcs x kind d body r (Hsub _ (or_introl eq_refl)) Hch Hd).
  - intros bindings body Hcodes IH kinds C Hsub H.
    cbn [repr_expr_frag] in H. cbn [repr_expr].
    apply andb_true_iff in H as [Hbs Hb].
    apply andb_true_iff. split.
    + rewrite forallb_forall in Hbs |- *. intros b Hbin.
      specialize (Hbs b Hbin).
      destruct b as [id c | id | s const];
        cbn [repr_binding_frag] in Hbs; cbn [repr_binding]; [ | exact Hbs
                                                          | exact Hbs ].
      destruct c as [params mc ret exn md cbody].
      cbn [repr_code_frag] in Hbs. cbn [repr_code].
      destruct (code_change_u sol cannot_change code_deps unit_code_ids
                  unboxed id md) as [ | pd rd];
        [ split_bools; discriminate | ].
      split_bools; try assumption.
      refine (Hcodes id (Code params mc ret exn md cbody) Hbin _ _ _ _);
        [ | eassumption ].
      intros p Hp. apply Hsub. cbn [let_bindings]. apply in_or_app. left.
      apply in_flat_map.
      exists (Bind_code id (Code params mc ret exn md cbody)).
      split; [ exact Hbin | exact Hp ].
    + refine (IH kinds C _ Hb). intros p Hp. apply Hsub.
      cbn [let_bindings]. apply in_or_app. right. exact Hp.
  - intros k h body IHh IH kinds C Hsub H. destruct h as [params md handler].
    cbn [repr_expr_frag] in H. cbn [repr_expr].
    apply andb_true_iff in H as [Hh Hb].
    apply andb_true_iff. split.
    + refine (IHh kinds C _ _ Hh). intros p Hp. apply Hsub.
      cbn [let_bindings handler_let_bindings]. apply in_or_app. left.
      exact Hp.
    + refine (IH _ _ _ Hb). intros p Hp. apply Hsub.
      cbn [let_bindings]. apply in_or_app. right. exact Hp.
  - intros handlers body IHh IH kinds C Hsub H.
    cbn [repr_expr_frag] in H. cbn [repr_expr].
    apply andb_true_iff in H as [Hhs Hb].
    apply andb_true_iff. split.
    + rewrite forallb_forall in Hhs |- *. intros [k h] Hkh.
      specialize (Hhs (k, h) Hkh). pose proof (IHh k h Hkh) as IHkh.
      destruct h as [params md handler]. cbv beta iota in Hhs |- *.
      refine (IHkh _ _ _ _ Hhs). intros p Hp. apply Hsub.
      cbn [let_bindings]. apply in_or_app. left. apply in_flat_map.
      exists (k, Handler params md handler). split; [ exact Hkh | exact Hp ].
    + refine (IH _ _ _ Hb). intros p Hp. apply Hsub.
      cbn [let_bindings]. apply in_or_app. right. exact Hp.
  - intros ap kinds C _ H. exact H.
  - intros ac kinds C _ H. exact H.
  - intros scrutinee arms kinds C _ H. exact H.
  - intros kinds C _ H. exact H.
  - intros params md handler IH kinds C ds Hsub H.
    cbn [repr_handler_frag] in H. cbn [repr_handler].
    split_bools; try assumption.
    exact (IH _ _ Hsub ltac:(eassumption)).
Qed.

End Mono.

Lemma option_code_eqb_refl (o : option code_id) : option_code_eqb o o = true.
Proof. destruct o; [ apply Pos.eqb_refl | reflexivity ]. Qed.

Theorem repr_unboxing_frag_sound : repr_unboxing_frag_sound_statement.
Proof.
  intros nr is_local u e T cannot_change code_deps dominated to_unbox tcr
    Hflow Hwfu Honce Hleast Hdom H.
  unfold repr_unboxing_frag_nr in H. unfold repr_unboxing_nr.
  apply andb_true_iff in H as [H Huses]. rewrite Huses, andb_true_r.
  refine (proj1 (frag_mono _ _ _ _ _ _ _ _ _ _ (let_bindings e) _) e [] []
            (fun p Hp => Hp) H).
  intros x kind d body r Hx Hch Hfrag.
  destruct d as [s | p args | id vs | slot decls vs | w c | pf w c | mf mt c];
    cbn [changed_set_frag_ok] in Hfrag; try discriminate Hfrag.
  unfold changed_name, changed_of_node in Hch.
  pose proof (changed_slot_current is_local u e T dominated to_unbox tcr x slot
                decls vs r Hflow Hwfu Honce Hleast Hdom Hx Hch) as Hslot.
  pose proof (changed_decls_renamed is_local _ T dominated to_unbox tcr e x slot
                decls vs _ r Hx Hch) as Hren.
  pose proof (changed_leaves_distinct is_local _ T dominated to_unbox tcr e _ r
                Hch) as Hdist.
  cbn [changed_set_ok]. split_bools; try assumption.
  - rewrite Hslot. apply Pos.eqb_refl.
  - unfold current_slot. rewrite <- Hslot, Hren. apply option_code_eqb_refl.
Qed.

(* ======================================================================== *)
(* The refinement                                                           *)
(* ======================================================================== *)

Module RCP := FlambdaReaperUnboxingRefinesClosuresProofs.

Theorem rebuild_repr_frag_refines_cc : rebuild_repr_frag_refines_cc_statement.
Proof.
  intros is_local u e T cannot_change dominated to_unbox tcr pdc nr Hloc Hflow
    Hwfu Honce Hleast Hcc Hdom Hfrag Hreads.
  exact (RCP.rebuild_repr_fixed_refines_cc_never_returns Auto is_local u e T
           cannot_change dominated to_unbox tcr pdc nr Hloc Hflow Hcc
           (repr_unboxing_frag_sound nr is_local u e T cannot_change _
              dominated to_unbox tcr Hflow Hwfu Honce Hleast Hdom Hfrag)
           Hreads).
Qed.

Theorem rebuild_repr_frag_observably : rebuild_repr_frag_observably_statement.
Proof.
  intros is_local u e T cannot_change dominated to_unbox tcr pdc nr Hloc Hproj
    Hflow Hany Hres Hwfu Honce Hexits Hleast Hcc Hexact Hccc Hwidgets Hdom
    Hfrag Hreads.
  exact (RCP.rebuild_repr_fixed_observably_never_returns Auto is_local u e T
           cannot_change dominated to_unbox tcr pdc nr Hloc Hproj Hflow Hany
           Hres Hwfu Hexits Hcc Hexact Hccc Hwidgets
           (repr_unboxing_frag_sound nr is_local u e T cannot_change _
              dominated to_unbox tcr Hflow Hwfu Honce Hleast Hdom Hfrag)
           Hreads).
Qed.

Theorem rebuild_repr_frag_local : rebuild_repr_frag_local_statement.
Proof.
  intros is_local u e T cannot_change dominated to_unbox tcr pdc nr Hloc Hflow
    Hany Hres Hwfu Honce Hexits Hleast Hcc Hlocal Hexact Hccc Hwidgets Hdom
    Hfrag Hreads.
  exact (RCP.rebuild_repr_fixed_local_never_returns Auto is_local u e T
           cannot_change dominated to_unbox tcr pdc nr Hloc Hflow Hany Hres
           Hwfu Hexits Hcc Hlocal Hexact Hccc Hwidgets
           (repr_unboxing_frag_sound nr is_local u e T cannot_change _
              dominated to_unbox tcr Hflow Hwfu Honce Hleast Hdom Hfrag)
           Hreads).
Qed.
