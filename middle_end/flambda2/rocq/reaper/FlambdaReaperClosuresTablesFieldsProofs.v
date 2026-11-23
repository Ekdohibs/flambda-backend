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

(* The proofs of FlambdaReaperClosuresTablesFieldsSpec.v. Nothing here
   needs review.

   [reprs_in]: a representation of the fold of [changed_representations]
   is that of the set of an allocation of [tcr], given to the names that a
   closure of that set dominates, with the fields that [mk_unboxed_fields]
   makes of those read from the closures of the set ([gfoc_spec] of
   FlambdaReaperTypesProofs.v). A field that it keeps has a constructor
   at the allocation ([mk_field_source]), which can change its
   representation and so holds no field of a block ([tcr_not_crr]).

   [frag2_mono]: by induction on the unit, as
   FlambdaReaperClosuresTablesProofs.frag_mono. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesClosuresSpec FlambdaReaperClosuresTablesSpec
  FlambdaReaperClosuresTablesFieldsSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs FlambdaReaperTypesProofs
  FlambdaReaperTypesSpec FlambdaReaperTypesTraversalSpec
  FlambdaReaperTypesTraversalProofs FlambdaReaperUnboxingProofs
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperClosuresExactProofs
  FlambdaReaperRebuildProofs FlambdaReaperLeastProofs
  FlambdaReaperClosuresTablesProofs.
From OxCamlFlambda2 Require Import FlambdaReaperSolutionProofs.
From OxCamlFlambda2 Require FlambdaReaperUnboxingFuelSpec
  FlambdaReaperUnboxingFuelProofs.
Import ListNotations.

Module UF := FlambdaReaperUnboxingFuelSpec.
Module UFP := FlambdaReaperUnboxingFuelProofs.

Module TP := FlambdaReaperTypesProofs.
Module TTP := FlambdaReaperTypesTraversalProofs.
Module URP := FlambdaReaperUnboxingRefinesProofs.
Module FP := FlambdaReaperFlowProofs.
Module CE := FlambdaReaperClosuresExactProofs.
Module RP := FlambdaReaperRebuildProofs.
Module CTP := FlambdaReaperClosuresTablesProofs.
Module SP := FlambdaReaperSolutionProofs.

Ltac split_bools :=
  repeat match goal with
  | H : (_ && _) = true |- _ => apply andb_true_iff in H as [? ?]
  | |- (_ && _) = true => apply andb_true_iff; split
  end.

(* ======================================================================== *)
(* The tables                                                               *)
(* ======================================================================== *)

Section Tables.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Variable cannot_change : list node.
Variable max_unbox_size : nat.
Variable lambda_lifting : bool.
Variable dominated : list (node * node).
Variable to_unbox tcr : list node.

Hypothesis Htables :
  analysis_tables is_local F T cannot_change max_unbox_size lambda_lifting
    dominated to_unbox tcr
  = true.

Abbreviation R :=
  (unboxing_relations is_local F T cannot_change max_unbox_size
     lambda_lifting).

Lemma tables_dominated : check_dominated T dominated = true.
Proof. unfold analysis_tables in Htables. split_bools. assumption. Qed.

Lemma tables_tcr (x : node) :
  x ∈ tcr ->
  mem_node x (t_has_usage T) = true ∧ mem_node x (ut_crr R) = false
  ∧ mem_node x to_unbox = false.
Proof.
  intros Hx. unfold analysis_tables, check_to_change_representation_fixed,
    check_to_change_representation_with in Htables.
  apply andb_true_iff in Htables as [Ht _].
  apply andb_true_iff in Ht as [_ Ht].
  apply andb_true_iff in Ht as [Ht _].
  apply andb_true_iff in Ht as [_ Ht].
  rewrite forallb_forall in Ht. specialize (Ht x Hx).
  apply andb_true_iff in Ht as [Ht Htu]. apply andb_true_iff in Ht as [Hu Hc].
  apply negb_true_iff in Htu, Hc. auto.
Qed.

Lemma tables_to_unbox (x : node) :
  x ∈ to_unbox -> mem_node x (t_has_usage T) = true.
Proof.
  intros Hx. unfold analysis_tables, check_to_change_representation_fixed,
    check_to_change_representation_with, check_to_unbox_with in Htables.
  split_bools. match goal with
  | H : forallb _ to_unbox = true |- _ => rewrite forallb_forall in H;
                                          specialize (H x Hx)
  end.
  split_bools. assumption.
Qed.

Lemma tables_sets_whole : check_sets_whole F T to_unbox tcr = true.
Proof. unfold analysis_tables in Htables. split_bools. assumption. Qed.

(* An allocation of [tcr] holds no field of a block. *)
Lemma tcr_not_block (x from : node) (f : field) :
  x ∈ tcr -> (x, f, from) ∈ f_constructor F -> is_block_like_field f = false.
Proof.
  intros Hx Hc. destruct (is_block_like_field f) eqn:Hb; [ | reflexivity ].
  exfalso. destruct (tables_tcr x Hx) as (_ & Hcrr & _).
  assert (Hin : x ∈ ut_crr R).
  { cbn [unboxing_relations ut_crr]. apply FlambdaReaperLeastProofs.in_dedup.
    apply in_or_app. right. apply in_flat_map. exists (x, f, from).
    rewrite Hb. split; [ exact Hc | left; reflexivity ]. }
  apply SP.mem_node_In in Hin. rewrite Hin in Hcrr. discriminate Hcrr.
Qed.

End Tables.

(* ======================================================================== *)
(* The representations                                                      *)
(* ======================================================================== *)

(* A representation of the fold: that of the set of an allocation [x] of
   [tcr], given to a name that a closure of the set dominates. *)
Lemma reprs_in (F : facts) (T : tables) (cm : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node)
    (is_local : field -> bool) (tcr : list node) (next : positive) (c : node)
    (r : closure_repr) :
  (c, r)
    ∈ fst (changed_representations F T cm dominated to_unbox is_local tcr
             next) ->
  ∃ (x f0 : node) (g : function_slot) (next' : positive),
    x ∈ tcr ∧ (g, f0) ∈ set_function_slots F x ∧ (f0, c) ∈ dominated
    ∧ cr_fields r
      = fst (mk_unboxed_fields F T cm dominated to_unbox
               (UF.unboxing_depth T) x
               (get_fields_of_constructors F T is_local
                  (map snd (set_function_slots F x)))
               next').
Proof.
  rewrite UFP.changed_representations_depth.
  match goal with
  | |- context [List.fold_left ?h tcr _] => set (step := h)
  end.
  set (P := fun p : node * closure_repr =>
    ∃ (x f0 : node) (g : function_slot) (next' : positive),
      x ∈ tcr ∧ (g, f0) ∈ set_function_slots F x ∧ (f0, fst p) ∈ dominated
      ∧ cr_fields (snd p)
        = fst (mk_unboxed_fields F T cm dominated to_unbox
                 (UF.unboxing_depth T) x
                 (get_fields_of_constructors F T is_local
                    (map snd (set_function_slots F x)))
                 next')).
  assert (Hgen :
    ∀ (l : list node) (acc : list (node * closure_repr)) (seen : list node)
      (n : positive),
      (∀ y : node, y ∈ l -> y ∈ tcr) -> (∀ p, p ∈ acc -> P p) ->
      ∀ p, p ∈ fst (fst (List.fold_left step l (acc, seen, n))) -> P p).
  { induction l as [ | x l IH]; intros acc seen n Hl Hacc; [ exact Hacc | ].
    cbn [List.fold_left].
    destruct (step (acc, seen, n) x) as [[acc' seen'] n'] eqn:Hstep.
    apply IH; [ intros y Hy; apply Hl; right; exact Hy | ].
    assert (Hx : x ∈ tcr) by (apply Hl; left; reflexivity).
    intros p Hp.
    unfold step in Hstep. cbv beta iota in Hstep.
    destruct (mem_node x seen);
      [ injection Hstep as <- _ _; exact (Hacc p Hp) | ].
    destruct (set_function_slots F x) as [ | [fs0 f00] l0] eqn:Hl0;
      [ injection Hstep as <- _ _; exact (Hacc p Hp) | ].
    destruct (mk_unboxed_fields F T cm dominated to_unbox
                (UF.unboxing_depth T) x
                (get_fields_of_constructors F T is_local
                   (map snd ((fs0, f00) :: l0)))
                n)
      as [fields n1] eqn:Hmk.
    destruct (rename_slots (map fst ((fs0, f00) :: l0)) n1) as [fss n2].
    injection Hstep as <- _ _.
    apply in_app_or in Hp as [Hp | Hp]; [ exact (Hacc p Hp) | ].
    assert (Hpf : ∃ (fs : function_slot) (f : node),
              (fs, f) ∈ (fs0, f00) :: l0
              ∧ p ∈ flat_map
                      (fun '(a, c) =>
                         if node_eqb a f
                         then [(c, Closure_repr fields fss fs)]
                         else [])
                      dominated).
    { apply in_app_or in Hp as [Hp1 | Hp1].
      - exists fs0, f00. split; [ left; reflexivity | exact Hp1 ].
      - apply in_flat_map in Hp1 as ([fs f] & Hfs & Hp2). exists fs, f.
        split; [ right; exact Hfs | exact Hp2 ]. }
    clear Hp. destruct Hpf as (fs & f & Hfs & Hp).
    apply in_flat_map in Hp as ([a c'] & Hd & Hp3).
    destruct (node_eqb a f) eqn:Haf; [ | destruct Hp3 ].
    destruct Hp3 as [<- | []]. apply node_eqb_eq in Haf. subst a.
    exists x, f, fs, n. cbn [fst snd cr_fields]. rewrite Hl0.
    split; [ exact Hx | split; [ exact Hfs | split; [ exact Hd | ] ] ].
    rewrite Hmk. reflexivity. }
  intros Hin.
  destruct (List.fold_left step tcr ([], [], next)) as [[acc seen] n] eqn:Hf.
  assert (Hin' : (c, r) ∈ fst (fst (List.fold_left step tcr ([], [], next))))
    by (rewrite Hf; exact Hin).
  exact (Hgen tcr [] [] next (fun y H => H) (fun p H => match H with end)
           (c, r) Hin').
Qed.

(* A field that [mk_unboxed_fields] keeps is a field of data with a
   source. *)
Lemma mk_field_source (F : facts) (T : tables) (cm : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (fuel : nat)
    (block : node) :
  ∀ (fields : list (field * field_use)) (next : positive) (f : field)
    (u : unboxed_field),
    (f, u) ∈ fst (mk_unboxed_fields F T cm dominated to_unbox fuel block
                    fields next) ->
    is_data_field f = true ∧ get_single_field_source F T block f <> No_source.
Proof.
  induction fields as [ | [f' use] fields IH]; intros next f u Hin.
  - rewrite URP.mk_unboxed_fields_nil in Hin. destruct Hin.
  - rewrite URP.mk_unboxed_fields_cons in Hin.
    destruct (URP.field_step F T cm dominated to_unbox fuel block f' use next)
      as [u0 n0] eqn:Hs.
    destruct (mk_unboxed_fields F T cm dominated to_unbox fuel block fields
                n0)
      as [rest n1] eqn:Hr.
    cbn [fst] in Hin.
    assert (Hrest : (f, u) ∈ rest -> is_data_field f = true
                                     ∧ get_single_field_source F T block f
                                       <> No_source).
    { intros H. apply (IH n0 f u). rewrite Hr. exact H. }
    destruct u0 as [u0 | ]; [ | exact (Hrest Hin) ].
    destruct Hin as [[= <- _] | Hin]; [ | exact (Hrest Hin) ].
    unfold URP.field_step in Hs.
    destruct (is_data_field f') eqn:Hd; cbn [negb] in Hs;
      [ | discriminate Hs ].
    split; [ reflexivity | ].
    destruct (get_single_field_source F T block f') eqn:Hg;
      [ discriminate Hs | discriminate | discriminate ].
Qed.

(* A field with a source is held by the allocation. *)
Lemma source_constructor (F : facts) (T : tables) (block : node) (f : field) :
  get_single_field_source F T block f <> No_source ->
  ∃ from : node, (block, f, from) ∈ f_constructor F.
Proof.
  unfold get_single_field_source. intros H.
  destruct (existsb _ (f_constructor F)) eqn:Hex.
  - apply existsb_exists in Hex as ([[b f'] from] & Hin & Hb).
    apply andb_true_iff in Hb as [Hb _]. apply andb_true_iff in Hb as [Hb Hf].
    apply node_eqb_eq in Hb. apply field_eqb_eq in Hf. subst b f'.
    exists from. exact Hin.
  - match type of H with
    | context [Keyed.dedup node_eqb ?l] =>
      destruct l as [ | s0 l'] eqn:Hl
    end; [ exfalso; apply H; reflexivity | ].
    assert (Hs0 : s0 ∈ s0 :: l') by (left; reflexivity).
    rewrite <- Hl in Hs0.
    apply in_flat_map in Hs0 as ([[b f'] fs] & Hin & Hs).
    destruct (node_eqb b block && field_eqb f f') eqn:Hb; [ | destruct Hs ].
    apply andb_true_iff in Hb as [Hb Hf].
    apply node_eqb_eq in Hb. apply field_eqb_eq in Hf. subst b f'.
    exists fs. exact Hin.
Qed.

(* ======================================================================== *)
(* The fields of a representation of a [Let]                                *)
(* ======================================================================== *)

Section Let_fields.

Variable is_local : field -> bool.
Variable u : unit_interface.
Variable e : expr.
Variable T : tables.
Variable cannot_change : list node.
Variable max_unbox_size : nat.
Variable lambda_lifting : bool.
Variable dominated : list (node * node).
Variable to_unbox tcr : list node.
Variable x : variable.
Variable slot : function_slot.
Variable decls : list (function_slot * code_id).
Variable vs : list (value_slot * simple).
Variable r : closure_repr.

Abbreviation F := (traverse_unit Auto u e).
Abbreviation G := (graph_of_facts F).
Abbreviation sol := (solution_of_tables T).
Abbreviation fsf a g b := (g_constructor G a (Field_function_slot g) b).

Hypothesis Hflow : in_flow_fragment e = true.
Hypothesis Hwfu : well_formed_unit u e = true.
Hypothesis Honce : variables_bound_once e = true.
Hypothesis Hleast : least_solution is_local F T.
Hypothesis Htables :
  analysis_tables is_local F T cannot_change max_unbox_size lambda_lifting
    dominated to_unbox tcr
  = true.
Hypothesis Hx : (x, Closure_of_set slot decls vs) ∈ let_bindings e.
Hypothesis Hlk :
  Keyed.lookup node_eqb (Node_variable x)
    (unit_changed_representations is_local F T dominated to_unbox tcr e)
  = Some r.

Lemma closed_tables : closed is_local G sol.
Proof. exact (TP.least_closed_tables is_local F T (or_introl Hleast)). Qed.

Lemma sets_of_traversal :
  FlambdaReaperTypesTraversalSpec.function_slots_of_sets G.
Proof. exact (TTP.function_slots_of_sets_traversal u e Hflow Hwfu Honce). Qed.

(* A field of [r] is a value slot read from a closure of the set of an
   allocation [x0] of [tcr], which holds it, and of which [x] is a
   closure. *)
Lemma field_origin (f : field) (uf : unboxed_field) :
  (f, uf) ∈ cr_fields r ->
  ∃ (x0 : node) (w : value_slot) (use : field_use) (from : node),
    f = Field_value_slot w ∧ x0 ∈ tcr
    ∧ (f, use)
      ∈ get_fields_of_constructors F T is_local
          (map snd (set_function_slots F x0))
    ∧ (x0, f, from) ∈ f_constructor F
    ∧ ∃ g : function_slot, fsf x0 g (Node_variable x).
Proof.
  intros Hfu.
  pose proof (CTP.closure_let_self u e x slot decls vs Hflow Hwfu Hx) as Hxx.
  unfold unit_changed_representations in Hlk.
  apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hlk.
  destruct (reprs_in _ _ _ _ _ _ _ _ _ _ Hlk)
    as (x0 & f0 & g & n' & Hx0 & Hg & Hd & Hfields).
  rewrite (TTP.dominated_self is_local F T dominated closed_tables
             (tables_dominated _ _ _ _ _ _ _ _ _ Htables) f0 (Node_variable x)
             (Node_variable x) slot Hxx Hd)
    in Hg.
  rewrite Hfields in Hfu.
  destruct (mk_field_source _ _ _ _ _ _ _ _ _ _ _ Hfu) as [Hdata Hsrc].
  destruct (source_constructor _ _ _ _ Hsrc) as (from & Hfrom).
  pose proof (tcr_not_block _ _ _ _ _ _ _ _ _ Htables x0 from f Hx0 Hfrom)
    as Hnb.
  destruct (URP.mk_unboxed_fields_in _ _ _ _ _ _ _ _ _ _ _ Hfu)
    as (use & Huse).
  destruct f as [i k | w | fs | | | | cw | rc | ];
    try discriminate Hdata; try discriminate Hnb.
  exists x0, w, use, from.
  split; [ reflexivity | split; [ exact Hx0 | split; [ exact Huse | ] ] ].
  split; [ exact Hfrom | ].
  exists g. exact (TP.set_function_slots_in F x0 g _ Hg).
Qed.

(* The closures of the set of [x0] are those of [x]. *)
Lemma closure_of_x (x0 c : node) (g : function_slot) :
  fsf x0 g (Node_variable x) -> c ∈ map snd (set_function_slots F x0) ->
  ∃ g' : function_slot, fsf (Node_variable x) g' c.
Proof.
  intros Hx0x Hc. apply in_map_iff in Hc as ([g' c'] & Ec & Hc).
  cbn [snd] in Ec. subst c'.
  pose proof (TP.set_function_slots_in F x0 g' c Hc) as Hx0c.
  exists g'.
  exact (proj1 ((proj2 (proj2 sets_of_traversal)) x0 (Node_variable x) c g
                  g' Hx0x) Hx0c).
Qed.

Lemma in_set_closure_nodes (c : node) (g : function_slot) :
  fsf (Node_variable x) g c -> c ∈ set_closure_nodes F (Node_variable x).
Proof.
  intros H. unfold set_closure_nodes. right. apply in_flat_map.
  exists (Node_variable x, Field_function_slot g, c).
  split; [ exact H | ]. rewrite node_eqb_refl. cbn. left. reflexivity.
Qed.

End Let_fields.

Theorem changed_field_used : changed_field_used_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr x slot decls vs r [f uf] Hflow Hwfu Honce Hleast Htables Hx
    Hlk Hfu.
  destruct (field_origin is_local u e T cannot_change max_unbox_size
              lambda_lifting dominated to_unbox tcr x slot decls vs r Hflow
              Hwfu Hleast Htables Hx Hlk f uf Hfu)
    as (x0 & w & use & from & -> & Hx0 & Huse & _ & g & Hx0x).
  exists w, uf. split; [ reflexivity | ].
  destruct (TP.gfoc_used is_local _ T [] (or_introl Hleast) _ _ _ Huse)
    as (c & Hc & Hused).
  destruct (closure_of_x u e x Hflow Hwfu Honce x0 c g Hx0x Hc) as (g' & Hxc).
  apply existsb_exists. exists c.
  split; [ exact (in_set_closure_nodes u e x c g' Hxc) | exact Hused ].
Qed.

(* A field with a use at a closure of [cl]: what the closure holds there
   has a use. *)
Lemma gfoc_holds_used (is_local : field -> bool) (F : facts) (T : tables)
    (cl : list node) (f : field) (use : field_use) :
  least_solution is_local F T ->
  (f, use) ∈ get_fields_of_constructors F T is_local cl ->
  ∃ c : node,
    c ∈ cl
    ∧ ∀ from : node,
        g_constructor (graph_of_facts F) c f from ->
        s_has_usage (solution_of_tables T) from = true.
Proof.
  intros Hleast Huse.
  pose proof (TP.least_closed_tables is_local F T (or_introl Hleast)) as Hc.
  destruct (TP.gfoc_spec is_local F T [] cl f use Huse) as [_ Hu].
  destruct use as [ | flow_to].
  - destruct Hu as (c & Hcl & Htop). exists c. split; [ exact Hcl | ].
    intros from Hcons. apply (has_usage_any _ _ _ Hc).
    exact (TP.field_used_top_any_usage is_local F T (or_introl Hleast) [] c
             from f Htop Hcons).
  - destruct Hu as (Hne & Hnot & Hiff).
    destruct flow_to as [ | y ys]; [ exfalso; apply Hne; reflexivity | ].
    destruct (proj1 (Hiff y) (or_introl eq_refl)) as (c & Hcl & Hy).
    exists c. split; [ exact Hcl | ].
    intros from Hcons.
    destruct (TP.field_used_as_alias is_local F T (or_introl Hleast) [] c
                from y f Hy Hcons (Hnot c Hcl))
      as (Hhu & _ & Hal).
    exact (has_usage_alias _ _ _ Hc y from Hhu Hal).
Qed.

Theorem changed_field_kept : changed_field_kept_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr x slot decls vs r w uf a Hflow Hwfu Honce Hleast Htables Hx
    Hlk Hfu Hw.
  set (F := traverse_unit Auto u e) in *.
  destruct (field_origin is_local u e T cannot_change max_unbox_size
              lambda_lifting dominated to_unbox tcr x slot decls vs r Hflow
              Hwfu Hleast Htables Hx Hlk _ uf Hfu)
    as (x0 & w' & use & from & Ew & Hx0 & Huse & Hfrom & g & Hx0x).
  injection Ew as <-.
  destruct (gfoc_holds_used is_local F T _ _ use Hleast Huse)
    as (c & Hc & Hcu).
  destruct (closure_of_x u e x Hflow Hwfu Honce x0 c g Hx0x Hc) as (g' & Hxc).
  (* The allocation of [x] holds exactly the facts of its set. *)
  assert (Hold : CE.no_old_sets e).
  { pose proof Hflow as Hf. unfold in_flow_fragment in Hf.
    apply andb_true_iff in Hf as [Hf _].
    exact (proj1 (CE.fragment_no_old_sets _ _ _) e Hf). }
  assert (Hnl : NoDup (map fst (let_bindings e))).
  { pose proof Honce as Ho. unfold variables_bound_once in Ho.
    apply distinct_NoDup in Ho. exact (NoDup_app_remove_r _ _ Ho). }
  assert (Hns : NoDup (map fst (CE.sym_bindings e))).
  { rewrite (proj1 CE.sym_bindings_static_names e Hold).
    apply distinct_NoDup. pose proof Hwfu as Hwf.
    unfold well_formed_unit in Hwf.
    repeat (apply andb_true_iff in Hwf as [Hwf ?]).
    assumption. }
  destruct (CE.traverse_unit_groups Auto u e Hnl Hns Hold)
    as (Gs & acc & HF & Hinv & _).
  pose proof (CE.closure_node_exact e Hnl Gs acc Hinv x slot decls vs Hx)
    as Hex.
  fold F in HF. rewrite <- HF in Hex.
  destruct Hex as (E1 & _ & E3 & E4).
  pose proof (CTP.closure_let_self u e x slot decls vs Hflow Hwfu Hx) as Hxx.
  destruct (TTP.function_slots_mutual_traversal u e Hflow Hwfu Honce x0
              (Node_variable x) g Hx0x)
    as (g0 & Hxx0).
  pose proof (proj1 (proj2 (E3 g0 x0 Hxx0) w from) Hfrom) as Hxw.
  destruct (E4 w from Hxw) as [Hkeep _].
  (* The edges of the [Let]: [x] holds at [w] what [vs] has there. *)
  destruct (proj1 (is_bound_lookup slot decls) (E1 slot _ Hxx)) as (id0 & Hl).
  destruct (FP.traverse_unit_facts Auto u e True Hflow (or_introl I))
    as (noted & deps & cds & Hedges & _).
  destruct (proj1 (CTP.let_edges _ noted deps cds _ _ _ True False) e None
              (unit_conts u) Hedges x _ Hx)
    as (ctx & Hset).
  cbn [FP.named_edges] in Hset.
  destruct (Hset id0 Hl) as (sites & Hsx & Hall).
  destruct (Hall slot id0 Hl) as (_ & _ & Hvs).
  destruct (Hvs w a Hw Hkeep) as [Ha _]. rewrite Hsx in Ha.
  pose proof (proj2 (proj2 (E3 g' c Hxc) w (node_of_simple a)) Ha) as Hca.
  pose proof (Hcu _ Hca) as Hu.
  destruct a as [z | sym | k]; cbn [kept_name node_of_simple] in *;
    [ exact Hu | exact Hu | reflexivity ].
Qed.

(* ======================================================================== *)
(* No name is unboxed and changed                                           *)
(* ======================================================================== *)

Theorem unboxed_changed_disjoint : unboxed_changed_disjoint_statement.
Proof.
  intros is_local F T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr e n Htables Hub.
  destruct (Keyed.lookup node_eqb n
              (unit_changed_representations is_local F T dominated to_unbox
                 tcr e))
    as [r | ] eqn:Hch; [ exfalso | reflexivity ].
  destruct (Keyed.lookup node_eqb n
              (fst (unit_unboxed_fields F T dominated to_unbox e)))
    as [fs | ] eqn:Hfs; [ | exact (Hub eq_refl) ].
  unfold unit_unboxed_fields in Hfs.
  apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hfs.
  destruct (URP.unboxed_fields_of_names_dom _ _ _ _ _ _ _ _ Hfs)
    as (a & _ & Had & Ha & _).
  unfold unit_changed_representations in Hch.
  apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hch.
  destruct (reprs_in _ _ _ _ _ _ _ _ _ _ Hch)
    as (x0 & f0 & g & _ & Hx0 & Hg & Hd & _).
  pose proof (tables_dominated _ _ _ _ _ _ _ _ _ Htables) as Hdom.
  destruct (proj1 (FlambdaReaperUnboxingProofs.check_dominated_sound T
                     dominated Hdom a n)
              (proj2 (mem_pair_In a n dominated) Had))
    as (Hsa & Hha & Hnm).
  destruct (proj1 (FlambdaReaperUnboxingProofs.check_dominated_sound T
                     dominated Hdom f0 n)
              (proj2 (mem_pair_In f0 n dominated) Hd))
    as (Hsf & Hhf & _).
  assert (Ea : a = f0).
  { destruct (node_eq_dec a f0) as [E | Hne]; [ exact E | ].
    exfalso. apply Hnm. right. exists a, f0. auto 6. }
  subst f0.
  apply mem_node_In in Ha.
  pose proof (tables_to_unbox _ _ _ _ _ _ _ _ _ Htables a Ha) as Hua.
  destruct (tables_tcr _ _ _ _ _ _ _ _ _ Htables x0 Hx0)
    as (Hux & _ & Hnx).
  pose proof (TP.set_function_slots_in F x0 g a Hg) as Hc.
  pose proof (tables_sets_whole _ _ _ _ _ _ _ _ _ Htables) as Hw.
  unfold check_sets_whole in Hw. rewrite forallb_forall in Hw.
  specialize (Hw _ Hc). cbn [is_function_slot negb orb] in Hw.
  rewrite Hux, Hua in Hw. cbn [negb orb] in Hw.
  apply andb_true_iff in Hw as [Hw _].
  rewrite Hnx, (proj2 (mem_node_In a to_unbox) Ha) in Hw. discriminate Hw.
Qed.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Section Mono2.

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

(* The conjuncts that [changed_set_frag2_ok] drops hold of the [Let]s
   among [L0]. *)
Hypothesis Hcs : ∀ (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (r : closure_repr),
  (x, d) ∈ L0 -> changed_name changed (Node_variable x) = Some r ->
  changed_set_frag2_ok unboxed changed x kind d body r = true ->
  changed_set_frag_ok F sol unboxed changed x kind d body r = true.

Abbreviation frag kinds C e :=
  (repr_expr_with F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr (changed_set_frag2_ok unboxed changed) kinds C e).
Abbreviation full kinds C e :=
  (repr_expr_frag F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr kinds C e).
Abbreviation frag_h kinds C ds h :=
  (repr_handler_with F sol cannot_change code_deps unit_code_ids codes
     unboxed changed my_closures nr (changed_set_frag2_ok unboxed changed)
     kinds C ds h).
Abbreviation full_h kinds C ds h :=
  (repr_handler_frag F sol cannot_change code_deps unit_code_ids codes
     unboxed changed my_closures nr kinds C ds h).

Definition mono2_P (e : expr) : Prop :=
  ∀ (kinds : finite_map flambda_kind) (C : finite_map (list decision * bool)),
    (∀ p, p ∈ let_bindings e -> p ∈ L0) ->
    frag kinds C e = true -> full kinds C e = true.

Lemma frag2_mono :
  (∀ e : expr, mono2_P e)
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
  apply FP.expr_handler_ind; unfold mono2_P.
  - intros x kind d body IH kinds C Hsub H.
    cbn [repr_expr_with] in H. cbn [repr_expr_frag].
    apply andb_true_iff in H as [H Hb]. apply andb_true_iff in H as [Hk Hd].
    rewrite Hk, (IH _ C (fun p Hp => Hsub p (or_intror Hp)) Hb).
    rewrite andb_true_r. cbn [andb].
    destruct (fields_of unboxed x) as [fields | ];
      destruct (changed_name changed (Node_variable x)) as [r | ] eqn:Hch;
      try exact Hd.
    exact (Hcs x kind d body r (Hsub _ (or_introl eq_refl)) Hch Hd).
  - intros bindings body Hcodes IH kinds C Hsub H.
    cbn [repr_expr_with] in H. cbn [repr_expr_frag].
    apply andb_true_iff in H as [Hbs Hb].
    apply andb_true_iff. split.
    + rewrite forallb_forall in Hbs |- *. intros b Hbin.
      specialize (Hbs b Hbin).
      destruct b as [id c | id | s const];
        cbn [repr_binding_with] in Hbs; cbn [repr_binding_frag]; [ | exact Hbs
                                                          | exact Hbs ].
      destruct c as [params mc ret exn md cbody].
      cbn [repr_code_with] in Hbs. cbn [repr_code_frag].
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
    cbn [repr_expr_with] in H. cbn [repr_expr_frag].
    apply andb_true_iff in H as [Hh Hb].
    apply andb_true_iff. split.
    + refine (IHh kinds C _ _ Hh). intros p Hp. apply Hsub.
      cbn [let_bindings handler_let_bindings]. apply in_or_app. left.
      exact Hp.
    + refine (IH _ _ _ Hb). intros p Hp. apply Hsub.
      cbn [let_bindings]. apply in_or_app. right. exact Hp.
  - intros handlers body IHh IH kinds C Hsub H.
    cbn [repr_expr_with] in H. cbn [repr_expr_frag].
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
    cbn [repr_handler_with] in H. cbn [repr_handler_frag].
    split_bools; try assumption.
    exact (IH _ _ Hsub ltac:(eassumption)).
Qed.

End Mono2.
Theorem repr_unboxing_frag2_sound : repr_unboxing_frag2_sound_statement.
Proof.
  intros nr is_local u e T cannot_change max_unbox_size lambda_lifting
    code_deps dominated to_unbox tcr Hflow Hwfu Honce Hleast Htables H.
  unfold repr_unboxing_frag2_nr in H. unfold repr_unboxing_frag_nr.
  apply andb_true_iff in H as [H Huses]. rewrite Huses, andb_true_r.
  refine (proj1 (frag2_mono _ _ _ _ _ _ _ _ _ _ (let_bindings e) _) e [] []
            (fun p Hp => Hp) H).
  intros x kind d body r Hx Hch Hfrag.
  destruct d as [s | p args | id vs | slot decls vs | w c | pf w c | mf mt c];
    cbn [changed_set_frag2_ok] in Hfrag; try discriminate Hfrag.
  unfold changed_name, changed_of_node in Hch.
  cbn [changed_set_frag_ok].
  apply andb_true_iff in Hfrag as [Hfrag Hfields].
  rewrite Hfrag. cbn [andb].
  rewrite forallb_forall in Hfields |- *. intros [f uf] Hfu.
  specialize (Hfields _ Hfu).
  destruct (changed_field_used is_local u e T cannot_change max_unbox_size
              lambda_lifting dominated to_unbox tcr x slot decls vs r (f, uf)
              Hflow Hwfu Honce Hleast Htables Hx Hch Hfu)
    as (w & uf' & Efu & Hused).
  injection Efu as -> <-.
  cbn [repr_field_frag_ok] in Hfields. cbn [repr_field_ok].
  rewrite Hused. cbn [andb].
  destruct (lookup w vs) as [a | ] eqn:Hw; [ | discriminate Hfields ].
  pose proof (changed_field_kept is_local u e T cannot_change max_unbox_size
                lambda_lifting dominated to_unbox tcr x slot decls vs r w uf a
                Hflow Hwfu Honce Hleast Htables Hx Hch Hfu Hw)
    as Hk.
  destruct uf as [v | nested]; [ rewrite Hfields, Hk; reflexivity | ].
  destruct a as [b | sym | k]; try discriminate Hfields.
  destruct (fields_of _ b) as [fb | ] eqn:Hfb; [ | discriminate Hfields ].
  cbn [kept_name] in Hk. rewrite Hk, Hfields. cbn [andb].
  unfold not_changed, changed_name, changed_of_node.
  rewrite (unboxed_changed_disjoint is_local _ T cannot_change max_unbox_size
             lambda_lifting dominated to_unbox tcr e (Node_variable b) Htables).
  - reflexivity.
  - unfold fields_of in Hfb. rewrite Hfb. discriminate.
Qed.

(* ======================================================================== *)
(* The refinement                                                           *)
(* ======================================================================== *)

Theorem rebuild_repr_frag2_refines_cc :
  rebuild_repr_frag2_refines_cc_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr pdc nr Hloc Hflow Hwfu Honce Hleast Hcc Htables Hfrag Hreads.
  exact (CTP.rebuild_repr_frag_refines_cc is_local u e T cannot_change
           dominated to_unbox tcr pdc nr Hloc Hflow Hwfu Honce Hleast Hcc
           (tables_dominated _ _ _ _ _ _ _ _ _ Htables)
           (repr_unboxing_frag2_sound nr is_local u e T cannot_change
              max_unbox_size lambda_lifting _ dominated to_unbox tcr Hflow
              Hwfu Honce Hleast Htables Hfrag)
           Hreads).
Qed.

Theorem rebuild_repr_frag2_observably :
  rebuild_repr_frag2_observably_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr pdc nr Hloc Hproj Hflow Hany Hres Hwfu Honce Hexits Hleast
    Hcc Hexact Hccc Hwidgets Htables Hfrag Hreads.
  exact (CTP.rebuild_repr_frag_observably is_local u e T cannot_change
           dominated to_unbox tcr pdc nr Hloc Hproj Hflow Hany Hres Hwfu Honce
           Hexits Hleast Hcc Hexact Hccc Hwidgets
           (tables_dominated _ _ _ _ _ _ _ _ _ Htables)
           (repr_unboxing_frag2_sound nr is_local u e T cannot_change
              max_unbox_size lambda_lifting _ dominated to_unbox tcr Hflow
              Hwfu Honce Hleast Htables Hfrag)
           Hreads).
Qed.

Theorem rebuild_repr_frag2_local : rebuild_repr_frag2_local_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr pdc nr Hloc Hflow Hany Hres Hwfu Honce Hexits Hleast Hcc
    Hlocal Hexact Hccc Hwidgets Htables Hfrag Hreads.
  exact (CTP.rebuild_repr_frag_local is_local u e T cannot_change dominated
           to_unbox tcr pdc nr Hloc Hflow Hany Hres Hwfu Honce Hexits Hleast
           Hcc Hlocal Hexact Hccc Hwidgets
           (tables_dominated _ _ _ _ _ _ _ _ _ Htables)
           (repr_unboxing_frag2_sound nr is_local u e T cannot_change
              max_unbox_size lambda_lifting _ dominated to_unbox tcr Hflow
              Hwfu Honce Hleast Htables Hfrag)
           Hreads).
Qed.

(* ======================================================================== *)
(* The examples                                                             *)
(* ======================================================================== *)

Theorem changed_in_slot_counterexample :
  changed_in_slot_counterexample_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100).
    vm_compute. reflexivity. }
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; vm_compute; reflexivity.
Qed.

Theorem cyclic_in_slot_alike : cyclic_in_slot_alike_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { apply (FlambdaReaperLeastProofs.check_least_sound _ _ 200).
    vm_compute. reflexivity. }
  do 5 (split; [ vm_compute; reflexivity | ]).
  split; [ vm_compute; discriminate | ].
  vm_compute. reflexivity.
Qed.
