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

(* The proofs of FlambdaReaperTypesTraversalUnaliasedSpec.v. Nothing here
   needs review.

   The facts of a unit put something ([tgt]) into a node held at a function
   slot only as [closures_unaliased] lets them. The traversal keeps
   [jinv]:
   - [fact_ok]: a fact puts something into the variable of a [Let] only
     from its binding in [e0] (nothing for a closure, its read for a
     projection of a function slot), into a parameter anything; into a
     symbol only from a binding of a constant that is no closure, or as
     imported. The steps add such facts ([po_ok], whose lemmas follow the
     [Kinds.ko_ok] of FlambdaReaperTypesTraversalProofs.v).
   - the made nodes held at a function slot ([cm]) are put into by no
     fact, and are none of the nodes of the code deps, the call witnesses
     of the apply deps and the continuations in scope ([jenv]): the steps
     of [KnownWitnesses] give it with the accumulator [UC.ghost] whose
     witnesses of known arity are [cm], as in
     FlambdaReaperUnusedCodeTraversalProofs.v; [prepare_code] with that
     of FlambdaReaperLeastProofs.AutoWidgets whose widgets are [cm]
     ([ghost_w]). The steps that add slot facts ([jstep_slots]) make
     nodes held at a function slot only fresh.
   - [held_ok]: a variable held at a function slot is that of a [Let] of a
     closure, or that of a projection of the slot off the variable of a
     [Let] that holds it there; a symbol is of a static closure. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperTypesTraversalSpec
  FlambdaReaperTypesTraversalUnaliasedSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs FlambdaReaperLeastProofs
  FlambdaReaperWrapperAutoProofs FlambdaReaperUnusedCodeTraversalProofs
  FlambdaReaperClosuresExactProofs FlambdaReaperTypesTraversalProofs.
Import ListNotations.

Module CE := FlambdaReaperClosuresExactProofs.
Module KW := FlambdaReaperWrapperAutoProofs.KnownWitnesses.
Module AW := FlambdaReaperLeastProofs.AutoWidgets.
Module UC := FlambdaReaperUnusedCodeTraversalProofs.
Module TT := FlambdaReaperTypesTraversalProofs.

Abbreviation conts := FlambdaReaperTraversal.env_conts.

(* ======================================================================== *)
(* What a fact puts something into                                          *)
(* ======================================================================== *)

(* The nodes that a fact puts something into, as [closures_unaliased]
   reads them. *)
Definition tgt (fct : fact) : list node :=
  match fct with
  | Fact_alias to_ _ | Fact_use to_ _ | Fact_accessor to_ _ _
  | Fact_parameter _ _ to_ | Fact_propagate _ to_ _
  | Fact_alias_if_any_source _ to_ _ => [to_]
  | Fact_any_source x | Fact_any_source_if_used _ x => [x]
  | Fact_constructor _ _ _ | Fact_argument _ _ _ | Fact_any_usage _ => []
  end.

Lemma tgt_into (fct : fact) (x : node) :
  x ∈ tgt fct -> x ∈ AW.into_nodes fct.
Proof. destruct fct; cbn; tauto. Qed.

(* The made nodes held at a function slot. *)
Definition cm_of (fct : fact) : list node :=
  match fct with
  | Fact_constructor _ (Field_function_slot _) (Node_made k) => [Node_made k]
  | _ => []
  end.

Definition cm (a : acc) : list node := flat_map cm_of (acc_facts a).

Lemma cm_of_in (fct : fact) (x : node) :
  x ∈ cm_of fct ->
  ∃ y g k, fct = Fact_constructor y (Field_function_slot g) (Node_made k)
           ∧ x = Node_made k.
Proof.
  destruct fct as [ | | | y f b | | | | | | | ]; cbn; try tauto.
  destruct f; cbn; try tauto. destruct b; cbn; try tauto.
  intros [<- | []]. eauto.
Qed.

Lemma cm_in (a : acc) (x : node) :
  x ∈ cm a
  <-> ∃ y g k, x = Node_made k
       ∧ Fact_constructor y (Field_function_slot g) (Node_made k)
         ∈ acc_facts a.
Proof.
  unfold cm. rewrite in_flat_map. split.
  - intros (fct & Hin & Hx).
    destruct (cm_of_in fct x Hx) as (y & g & k & -> & ->). eauto 6.
  - intros (y & g & k & -> & Hin).
    exists (Fact_constructor y (Field_function_slot g) (Node_made k)).
    split; [ exact Hin | left; reflexivity ].
Qed.

Lemma cm_made (a : acc) (x : node) : x ∈ cm a -> ∃ k, x = Node_made k.
Proof. intros H. apply cm_in in H as (y & g & k & -> & _). eauto. Qed.

Lemma cm_ns (a a' : acc) (x : node) :
  CE.ns_ok a a' -> x ∈ cm a' <-> x ∈ cm a.
Proof.
  intros (H1 & H2 & _). rewrite !cm_in.
  split; intros (y & g & k & -> & Hin); exists y, g, k; split; auto.
  apply H2; [ exact Hin | cbn; reflexivity ].
Qed.

Lemma cm_app (L : list fact) (a a' : acc) (x : node) :
  acc_facts a' = L ++ acc_facts a -> x ∈ cm a' ->
  x ∈ flat_map cm_of L ∨ x ∈ cm a.
Proof.
  intros E H. unfold cm in *. rewrite E, flat_map_app in H.
  apply in_app_or. exact H.
Qed.

Definition is_closure_const (c : static_const) : bool :=
  match c with
  | Static_const_closure_of_set _ _ _ _ => true
  | Static_const_set_of_closures _ _ | Static_const_block _ _
  | Static_const_boxed_number _ | Static_const_immutable_string _
  | Static_const_immutable_value_array _ => false
  end.

Section Facts.

Variable e0 : expr.

(* What the binding [d] of [v] may put into it: nothing for a set, its
   read for a projection of a function slot. *)
Definition named_tgt (v : variable) (d : named) (fct : fact) : Prop :=
  match d with
  | Set_of_closures _ _ | Closure_of_set _ _ _ => False
  | Project_function_slot _ o c =>
    fct = Fact_accessor (Node_variable v) (Field_function_slot o)
            (node_of_simple c)
  | Simple _ | Prim _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ => True
  end.

(* What [fct] may put into [x]. *)
Definition node_ok (fct : fact) (x : node) : Prop :=
  match x with
  | Node_variable v =>
    v ∈ param_variables e0
    ∨ ∃ d, (v, d) ∈ let_bindings e0 ∧ named_tgt v d fct
  | Node_symbol s =>
    (∃ c, (s, c) ∈ CE.sym_bindings e0 ∧ is_closure_const c = false)
    ∨ s ∉ static_names symbols_of_binding e0
  | Node_code_id _ | Node_all_constants | Node_made _ => True
  end.

Definition fact_ok (fct : fact) : Prop := ∀ x, x ∈ tgt fct -> node_ok fct x.

(* A node that anything may be put into: a parameter or a node of no
   variable and no symbol. *)
Definition pv (x : node) : Prop :=
  match x with
  | Node_variable v => v ∈ param_variables e0
  | Node_symbol _ => False
  | Node_code_id _ | Node_all_constants | Node_made _ => True
  end.

Lemma pv_node_ok (fct : fact) (x : node) : pv x -> node_ok fct x.
Proof. destruct x; cbn; tauto. Qed.

Lemma let_node_ok (fct : fact) (v : variable) (d : named) :
  (v, d) ∈ let_bindings e0 -> named_tgt v d fct ->
  node_ok fct (Node_variable v).
Proof. intros H1 H2. right. exists d. auto. Qed.

(* ======================================================================== *)
(* Steps that add facts of [fact_ok]                                        *)
(* ======================================================================== *)

Definition po_ok (a a' : acc) : Prop :=
  (∀ fct, fct ∈ acc_facts a' -> fct ∈ acc_facts a ∨ fact_ok fct)
  ∧ (∀ id cd, (id, cd) ∈ acc_code_deps a' ->
       (id, cd) ∈ acc_code_deps a ∨ ∀ x, x ∈ AW.code_dep_nodes cd -> pv x)
  ∧ (∀ d, d ∈ acc_apply_deps a' ->
       d ∈ acc_apply_deps a ∨ pv (ad_apply_call_witness d)).

Lemma po_refl (a : acc) : po_ok a a.
Proof. split; [ auto | split; auto ]. Qed.

Lemma po_trans (a a1 a2 : acc) : po_ok a a1 -> po_ok a1 a2 -> po_ok a a2.
Proof.
  intros (H1 & H2 & H3) (H4 & H5 & H6). split; [ | split ].
  - intros fct H. destruct (H4 fct H) as [H' | H']; [ | right; exact H' ].
    exact (H1 fct H').
  - intros id cd H. destruct (H5 id cd H) as [H' | H']; [ | right; exact H' ].
    exact (H2 id cd H').
  - intros d H. destruct (H6 d H) as [H' | H']; [ | right; exact H' ].
    exact (H3 d H').
Qed.

Lemma po_same (a a' : acc) :
  acc_facts a' = acc_facts a -> acc_code_deps a' = acc_code_deps a ->
  acc_apply_deps a' = acc_apply_deps a -> po_ok a a'.
Proof.
  intros E1 E2 E3. split; [ | split ].
  - intros fct H. left. rewrite <- E1. exact H.
  - intros id cd H. left. rewrite <- E2. exact H.
  - intros d H. left. rewrite <- E3. exact H.
Qed.

Lemma po_add_facts (L : list fact) (a : acc) :
  (∀ fct, fct ∈ L -> fact_ok fct) -> po_ok a (add_facts L a).
Proof.
  intros H. split; [ | split; [ intros id cd Hc; left; exact Hc
                              | intros d Hd; left; exact Hd ] ].
  intros fct Hin. cbn [add_facts acc_facts] in Hin.
  apply in_app_or in Hin as [Hin | Hin]; [ right; exact (H _ Hin)
                                        | left; exact Hin ].
Qed.

Lemma po_add_fact (fct : fact) (a : acc) :
  fact_ok fct -> po_ok a (add_fact fct a).
Proof. intros H. apply po_add_facts. intros fct' [<- | []]. exact H. Qed.

Lemma po_fresh (a : acc) : po_ok a (snd (fresh a)).
Proof. apply po_same; reflexivity. Qed.

Lemma po_fresh_list (n : nat) (a : acc) : po_ok a (snd (fresh_list n a)).
Proof.
  destruct (AW.fresh_list_spec n a) as (H1 & H2 & H3 & _).
  apply po_same; assumption.
Qed.

Lemma po_note_simple (s : simple) (a : acc) : po_ok a (note_simple s a).
Proof. destruct s; apply po_same; reflexivity. Qed.

Lemma po_note_simples (l : list simple) (a : acc) :
  po_ok a (note_simples l a).
Proof.
  destruct (AW.note_simples_fields l a) as (_ & H1 & H2 & H3 & _).
  apply po_same; assumption.
Qed.

Lemma po_push_apply_dep (d : apply_dep) (a : acc) :
  pv (ad_apply_call_witness d) -> po_ok a (push_apply_dep d a).
Proof.
  intros H. split; [ intros fct Hf; left; exact Hf | split ].
  - intros id cd Hc. left. exact Hc.
  - intros d' [<- | Hd]; [ right; exact H | left; exact Hd ].
Qed.

Lemma po_add_code_dep (id : code_id) (cd : code_dep) (a : acc) :
  (∀ x, x ∈ AW.code_dep_nodes cd -> pv x) -> po_ok a (add_code_dep id cd a).
Proof.
  intros H. split; [ intros fct Hf; left; exact Hf | split ].
  - intros id' cd' [Heq | Hc]; [ injection Heq as <- <-; right; exact H
                               | left; exact Hc ].
  - intros d Hd. left. exact Hd.
Qed.

Lemma po_fold {A : Type} (f : acc -> A -> acc) (l : list A) :
  (∀ (a : acc) (y : A), y ∈ l -> po_ok a (f a y)) ->
  ∀ a : acc, po_ok a (List.fold_left f l a).
Proof.
  induction l as [ | y l IH]; intros Hf a; [ apply po_refl | ].
  cbn [List.fold_left].
  apply (po_trans _ (f a y)); [ apply Hf; left; reflexivity | ].
  apply IH. intros a' y' Hy. apply Hf. right. exact Hy.
Qed.

(* Shows that a fact is of [fact_ok]. *)
Ltac po_node :=
  let x := fresh "x" in let Hx := fresh "Hx" in
  intros x Hx; cbn [tgt In] in Hx;
  repeat match type of Hx with _ ∨ _ => destruct Hx as [Hx | Hx] end;
  try contradiction; subst x; cbn [node_ok];
  first [ exact I | apply pv_node_ok; solve [ auto ] | idtac ].

(* Shows that the facts of a list are of [fact_ok]. *)
Ltac po_free :=
  let fct := fresh "fct" in let Hin := fresh "Hin" in
  intros fct Hin;
  repeat match goal with
  | H : In _ (_ ++ _) |- _ => apply in_app_or in H; destruct H as [H | H]
  | H : In _ (map _ _) |- _ =>
    let y := fresh "y" in let Heq := fresh "Heq" in let Hy := fresh "Hy" in
    apply in_map_iff in H; destruct H as [y [Heq Hy]]; subst fct;
    repeat match goal with
           | p : (_ * _)%type |- _ => destruct p
           end
  | H : In _ (_ :: _) |- _ => destruct H as [H | H]; [ subst fct | ]
  | H : In _ [] |- _ => destruct H
  end;
  po_node.

Ltac po_let H lem :=
  match type of H with
  | ?f = (?x, ?a1) =>
    let Hn := fresh "Hpo" in
    assert (Hn : po_ok _ a1)
      by (change a1 with (snd (x, a1)); rewrite <- H; apply lem)
  end.


(* The parameters of the continuations in scope may have anything put into
   them. *)
Definition penv (env : traverse_env) : Prop :=
  ∀ k ps x, (k, ps) ∈ conts env -> x ∈ ps -> pv x.

Lemma po_find_cont (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  penv env ->
  po_ok a (snd (find_cont env k n a))
  ∧ ∀ x, x ∈ fst (find_cont env k n a) -> pv x.
Proof.
  intros Henv. unfold find_cont.
  destruct (lookup k (conts env)) as [ps | ] eqn:E.
  - split; [ apply po_refl | ]. intros x Hx.
    exact (Henv k ps x (lookup_in _ _ _ E) Hx).
  - destruct (AW.fresh_list_spec n a) as (_ & _ & _ & _ & _ & Hm).
    destruct (fresh_list n a) as [ps b] eqn:Eb. cbn [fst snd] in *.
    po_let Eb po_fresh_list. split.
    + apply (po_trans _ _ _ Hpo). apply po_add_facts. po_free.
    + intros x Hx. destruct (Hm x Hx) as (m & -> & _). exact I.
Qed.

Lemma po_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  po_ok a (cond_any_usage env s a).
Proof.
  unfold cond_any_usage. apply (po_trans _ (note_simple s a));
    [ apply po_note_simple | ].
  destruct (env_current_code_id env); apply po_add_fact; po_node.
Qed.

Lemma po_cond_alias (env : traverse_env) (from to_ : node) (a : acc) :
  pv to_ -> po_ok a (cond_alias env from to_ a).
Proof.
  intros H. unfold cond_alias.
  destruct (env_current_code_id env); apply po_add_fact; po_node.
Qed.

Lemma po_block_deps (base : node) (fields : list simple) (a : acc) :
  po_ok a (block_deps base fields a).
Proof.
  unfold block_deps. apply (po_trans _ _ _ (po_note_simples fields a)).
  apply po_add_facts. po_free.
Qed.

Lemma po_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  (x, Prim p args) ∈ let_bindings e0 ->
  po_ok a (traverse_prim env x p args a).
Proof.
  intros Hx.
  assert (Hacc : ∀ (f : field) (s : simple) (a0 : acc),
             po_ok a0 (add_fact (Fact_accessor (Node_variable x) f
                                   (node_of_simple s)) (note_simple s a0))).
  { intros f s a0. apply (po_trans _ _ _ (po_note_simple s a0)).
    apply po_add_fact. po_node. exact (let_node_ok _ _ _ Hx I). }
  assert (Hdefault :
            po_ok a
              (add_facts
                 (Fact_any_source (Node_variable x)
                  :: map (fun from => Fact_use (Node_variable x) from)
                       (free_names_of_simples args))
                 (if prim_has_arbitrary_effects p
                  then cond_any_usage env (Var x) a else a))).
  { apply (po_trans _ (if prim_has_arbitrary_effects p
                        then cond_any_usage env (Var x) a else a)).
    - destruct (prim_has_arbitrary_effects p);
        [ apply po_cond_any_usage | apply po_refl ].
    - apply po_add_facts. po_free; exact (let_node_ok _ _ _ Hx I). }
  unfold traverse_prim; cbv zeta.
  destruct p as [u0 | b0 | v0 | h0 | r0];
    [ destruct u0 | | destruct v0 | destruct h0 | destruct r0 ];
    try (destruct args as [ | arg [ | arg2 args]]);
    cbv beta iota;
    try first [ exact Hdefault | apply po_block_deps | apply po_refl
              | exact (Hacc _ _ _) ].
  all: try (destruct variant_only;
            first [ exact Hdefault | exact (Hacc _ _ _) ]).
  - destruct mut; [ | exact (Hacc _ _ _) | exact (Hacc _ _ _) ].
    eapply po_trans; [ | apply po_add_fact; po_node ]; [ apply Hacc | ].
    exact (let_node_ok _ _ _ Hx I).
  - apply (po_trans _ _ _ (po_note_simple arg a)).
    apply po_add_fact. po_node.
Qed.

Lemma record_set_deps (env : traverse_env)
    (closures : list (node * option code_id))
    (fs : list (function_slot * node)) (vs : list (value_slot * simple)) :
  ∀ a : acc,
    acc_code_deps (record_set_of_closures_deps env closures fs vs a)
    = acc_code_deps a
    ∧ acc_apply_deps (record_set_of_closures_deps env closures fs vs a)
      = acc_apply_deps a.
Proof.
  unfold record_set_of_closures_deps.
  induction closures as [ | [base code] closures IH]; intros a;
    [ split; reflexivity | ].
  cbn [List.fold_left]. rewrite (proj1 (IH _)), (proj2 (IH _)).
  cbn [add_facts acc_code_deps acc_apply_deps].
  destruct (AW.note_simples_fields (map snd (filter (fun '(w, _) =>
              should_keep_value_slot env w) (compact vs)))
              (match code with
               | Some id => push_closure_dep
                              {| cld_let_bound_name_of_the_closure := base;
                                 cld_closure_code_id := id |} a
               | None => a
               end)) as (_ & _ & H3 & H4 & _).
  rewrite H3, H4. destruct code; split; reflexivity.
Qed.

Lemma po_record_set (env : traverse_env)
    (closures : list (node * option code_id))
    (fs : list (function_slot * node)) (vs : list (value_slot * simple))
    (a : acc) :
  po_ok a (record_set_of_closures_deps env closures fs vs a).
Proof.
  destruct (CE.record_set_facts env closures fs vs a) as [(L & HL & HLf) _].
  destruct (record_set_deps env closures fs vs a) as [Hc Hd].
  split; [ | split ].
  - intros fct Hin. rewrite HL in Hin.
    apply in_app_or in Hin as [Hin | Hin]; [ right | left; exact Hin ].
    apply HLf in Hin
      as (base & code & _ & [(w & s & _ & ->) | (f & c & _ & ->)]);
      intros x [].
  - intros id cd H. left. rewrite <- Hc. exact H.
  - intros d H. left. rewrite <- Hd. exact H.
Qed.

Lemma closures_of_set_deps (x : variable) (slot : function_slot)
    (others : list (function_slot * variable)) :
  ∀ (decls : list (function_slot * code_id)) (a : acc),
    acc_facts (snd (closures_of_set x slot others decls a)) = acc_facts a
    ∧ acc_code_deps (snd (closures_of_set x slot others decls a))
      = acc_code_deps a
    ∧ acc_apply_deps (snd (closures_of_set x slot others decls a))
      = acc_apply_deps a.
Proof.
  induction decls as [ | [f id] decls IH]; intros a;
    [ repeat split; reflexivity | ].
  cbn [closures_of_set].
  destruct (if Pos.eqb f slot then (Node_variable x, a)
            else match lookup f others with
                 | Some y => (Node_variable y, a)
                 | None => fresh a
                 end) as [name a1] eqn:E1.
  assert (H1 : acc_facts a1 = acc_facts a ∧ acc_code_deps a1 = acc_code_deps a
               ∧ acc_apply_deps a1 = acc_apply_deps a).
  { destruct (Pos.eqb f slot); [ injection E1 as _ <-; auto | ].
    destruct (lookup f others); [ injection E1 as _ <-; auto | ].
    unfold fresh in E1. injection E1 as _ <-. cbn. auto. }
  destruct (IH a1) as (H2 & H3 & H4).
  destruct (closures_of_set x slot others decls a1) as [rest a2].
  cbn [snd] in *. destruct H1 as (H5 & H6 & H7).
  rewrite H2, H3, H4. auto.
Qed.

Lemma po_traverse_named (env : traverse_env) (x : variable) (d : named)
    (a : acc) :
  (x, d) ∈ let_bindings e0 -> po_ok a (traverse_named env x d a).
Proof.
  intros Hx.
  destruct d as [s | p args | id vs | slot decls vs | w c | pf w c | mf mt c];
    cbn [traverse_named]; cbv zeta.
  - apply (po_trans _ _ _ (po_note_simple s a)). apply po_add_fact. po_node.
    exact (let_node_ok _ _ _ Hx I).
  - apply po_traverse_prim. exact Hx.
  - apply po_record_set.
  - unfold traverse_closure_of_set.
    destruct (among slot (map fst (compact decls))); [ | apply po_refl ].
    destruct (closures_of_set_deps x slot (set_closures_of env x)
                (compact decls) a) as (H1 & H2 & H3).
    destruct (closures_of_set x slot (set_closures_of env x) (compact decls) a)
      as [cl a1] eqn:Ecl. cbn [snd] in *.
    apply (po_trans _ a1); [ apply po_same; assumption | ].
    apply po_record_set.
  - apply (po_trans _ _ _ (po_note_simple c a)). apply po_add_fact. po_node.
    exact (let_node_ok _ _ _ Hx I).
  - apply (po_trans _ _ _ (po_note_simple c a)). apply po_add_fact. po_node.
    exact (let_node_ok _ _ _ Hx I).
  - apply (po_trans _ _ _ (po_note_simple c a)). apply po_add_fact. po_node.
    exact (let_node_ok _ _ _ Hx eq_refl).
Qed.

Lemma po_traverse_static_binding (env : traverse_env) (b : static_binding)
    (a : acc) :
  (∀ s c, b = Bind_symbol s c -> (s, c) ∈ CE.sym_bindings e0) ->
  po_ok a (traverse_static_binding env b a).
Proof.
  intros Hb. destruct b as [id c | id | s const];
    cbn [traverse_static_binding]; try apply po_refl.
  pose proof (Hb s const eq_refl) as Hs.
  destruct const as [id vs | slot cs decls vs | tag fields | contents
                    | bytes | [ | el els]].
  - apply po_record_set.
  - apply po_record_set.
  - apply po_block_deps.
  - apply po_add_facts. unfold boxed_number_facts.
    destruct (boxed_number_kind contents); [ po_free | ].
    destruct contents; po_free; left; eexists; split; eauto.
  - apply po_add_fact. po_node. left. eexists. split; eauto.
  - apply po_add_fact. po_node. left. eexists. split; eauto.
  - apply po_block_deps.
Qed.

Lemma po_add_facts_at (L : list fact) (a b : acc) :
  acc_facts b = acc_facts a -> acc_code_deps b = acc_code_deps a ->
  acc_apply_deps b = acc_apply_deps a ->
  (∀ fct, fct ∈ L -> fact_ok fct) -> po_ok a (add_facts L b).
Proof.
  intros E1 E2 E3 H. apply (po_trans _ b); [ apply po_same; assumption | ].
  apply po_add_facts. exact H.
Qed.

(* A node that the facts of a step put something into is of [pv], from
   [Hp]. *)
Ltac pv_from Hp :=
  apply pv_node_ok; apply Hp;
  first [ left; reflexivity
        | right; left; reflexivity
        | eapply AW.indexed_in; eassumption
        | right; eapply AW.indexed_in; eassumption
        | right; right; eapply AW.indexed_in; eassumption
        | eassumption ].

Lemma po_create_known (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  (∀ x, x ∈ params -> pv x) ->
  po_ok a (snd (create_known_arity_call_witness id params returns exn a)).
Proof.
  intros Hp. unfold create_known_arity_call_witness, fresh. cbn [fst snd].
  apply po_add_facts_at; try reflexivity. po_free. pv_from Hp.
Qed.

Lemma po_witness_deps (id : code_id) (returns : list node) (exn : node) :
  ∀ (l : list (list node * node)) (a : acc),
    (∀ first w x, (first, w) ∈ l -> x ∈ first -> pv x) ->
    po_ok a (unknown_arity_witness_deps id returns exn l a).
Proof.
  induction l as [ | [first w] rest IH]; intros a Hp; [ apply po_refl | ].
  cbn [unknown_arity_witness_deps].
  assert (Hf : ∀ x, x ∈ first -> pv x)
    by (intros x Hx; exact (Hp first w x (or_introl eq_refl) Hx)).
  assert (Hr : ∀ first' w' x, (first', w') ∈ rest -> x ∈ first' -> pv x)
    by (intros f' w' x H Hx; exact (Hp f' w' x (or_intror H) Hx)).
  match goal with
  | |- context [add_facts ?L a] =>
    apply (po_trans _ (add_facts L a));
      [ apply po_add_facts; po_free; pv_from Hf | ]
  end.
  destruct rest as [ | [? next] rest'].
  - apply po_add_facts. po_free.
  - unfold fresh. cbv beta iota.
    match goal with
    | |- po_ok ?b (unknown_arity_witness_deps _ _ _ _ (add_facts ?L ?c)) =>
      apply (po_trans _ (add_facts L c));
        [ apply po_add_facts_at; try reflexivity; po_free | ]
    end.
    apply IH. exact Hr.
Qed.

Lemma po_create_unknown (id : code_id) (is_tupled : bool) (ar : arity)
    (params returns : list node) (exn : node) (a : acc) :
  (∀ x, x ∈ params -> pv x) ->
  po_ok a
    (snd (create_unknown_arity_call_witnesses id is_tupled ar params returns
            exn a)).
Proof.
  intros Hp. unfold create_unknown_arity_call_witnesses. destruct is_tupled.
  - unfold create_unknown_arity_tupled_call_witnesses, fresh.
    cbn [fst snd]. apply po_add_facts_at; try reflexivity.
    po_free. pv_from Hp.
  - unfold create_unknown_arity_non_tupled_call_witnesses.
    match goal with
    | |- context [fresh_list ?n a] => destruct (fresh_list n a) as [ws a1] eqn:E
    end.
    po_let E po_fresh_list. cbn [snd]. apply (po_trans _ _ _ Hpo).
    apply po_witness_deps. intros first w x H Hx.
    apply Hp. apply in_combine_l in H. exact (AW.group_in _ _ _ _ H Hx).
Qed.

Lemma po_prepare_code (id : code_id) (c : code) (a : acc) :
  po_ok a (prepare_code id c a).
Proof.
  unfold prepare_code.
  destruct (AW.fresh_list_spec (length (code_result_kinds c)) a)
    as (_ & _ & _ & _ & _ & Hr).
  destruct (fresh_list _ a) as [returns a1] eqn:E1. po_let E1 po_fresh_list.
  cbn [fst] in Hr. unfold fresh at 1. cbv beta iota.
  unfold fresh at 1. cbv beta iota.
  match goal with
  | |- context [fresh_list ?n ?b] =>
    destruct (AW.fresh_list_spec n b) as (_ & _ & _ & _ & _ & Hq);
    destruct (fresh_list n b) as [params a4] eqn:E4
  end.
  po_let E4 po_fresh_list. cbn [fst] in Hq.
  assert (Hpv : ∀ x, x ∈ params -> pv x).
  { intros x Hx. destruct (Hq x Hx) as (m & -> & _). exact I. }
  destruct (create_known_arity_call_witness id params returns _ a4)
    as [known a5] eqn:E5.
  assert (Hpo5 : po_ok a4 a5)
    by (change a5 with (snd (known, a5)); rewrite <- E5;
        apply po_create_known; exact Hpv).
  destruct (create_unknown_arity_call_witnesses _ _ _ _ _ _ a5)
    as [unknown a6] eqn:E6.
  assert (Hpo6 : po_ok a5 a6)
    by (change a6 with (snd (unknown, a6)); rewrite <- E6;
        apply po_create_unknown; exact Hpv).
  apply (po_trans _ _ _ Hpo). eapply po_trans; [ apply po_same; reflexivity | ].
  eapply po_trans; [ apply po_same; reflexivity | ].
  apply (po_trans _ _ _ Hpo0), (po_trans _ _ _ Hpo5), (po_trans _ _ _ Hpo6).
  match goal with
  | |- context [add_fact ?f a6] =>
    apply (po_trans _ (add_fact f a6)); [ apply po_add_fact; po_node | ]
  end.
  apply po_add_code_dep. cbn [AW.code_dep_nodes cd_my_closure cd_exn
    cd_return]. intros x [<- | [<- | Hx]]; [ exact I | exact I | ].
  destruct (Hr x Hx) as (m & -> & _). exact I.
Qed.

Lemma po_prepare_binding (b : static_binding) (a : acc) :
  po_ok a (prepare_binding b a).
Proof.
  destruct b; [ apply po_prepare_code | apply po_refl | apply po_refl ].
Qed.

Lemma po_make_known (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  (∀ x, x ∈ exn :: returns -> pv x) ->
  po_ok a (snd (make_known_arity_apply_widget env ap returns exn a))
  ∧ pv (fst (make_known_arity_apply_widget env ap returns exn a)).
Proof.
  intros Hp. unfold make_known_arity_apply_widget, fresh.
  cbn [fst snd]. split; [ | exact I ].
  eapply po_trans; [ | apply po_cond_alias; exact I ].
  eapply po_trans; [ | apply po_add_facts; po_free; pv_from Hp ].
  eapply po_trans; [ | apply po_note_simples ].
  apply po_same; reflexivity.
Qed.

Lemma po_apply_deps (returns : list node) (exn called : node) :
  ∀ (l : list (list simple * node)) (a : acc),
    (∀ x, x ∈ exn :: called :: returns -> pv x) ->
    (∀ first w, (first, w) ∈ l -> pv w) ->
    po_ok a (unknown_arity_apply_deps returns exn called l a).
Proof.
  intros l. induction l as [ | [first w] rest IH]; intros a Hp Hw;
    [ apply po_refl | ].
  cbn [unknown_arity_apply_deps].
  assert (Hr : ∀ first' w', (first', w') ∈ rest -> pv w')
    by (intros f' w' H; exact (Hw f' w' (or_intror H))).
  match goal with
  | |- context [add_facts ?L a] =>
    apply (po_trans _ (add_facts L a));
      [ apply po_add_facts; po_free; pv_from Hp | ]
  end.
  destruct rest as [ | [? next] rest'].
  - apply po_add_facts. po_free.
    apply pv_node_ok, Hp. right. right. eapply AW.indexed_in. eassumption.
  - unfold fresh. cbv beta iota.
    match goal with
    | |- po_ok ?b (unknown_arity_apply_deps _ _ _ _ (add_facts ?L ?c)) =>
      apply (po_trans _ (add_facts L c));
        [ apply po_add_facts_at; try reflexivity; po_free | ]
    end.
    + apply pv_node_ok. exact (Hr _ _ (or_introl eq_refl)).
    + apply IH; assumption.
Qed.

Lemma po_make_unknown (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  (∀ x, x ∈ exn :: returns -> pv x) ->
  po_ok a (snd (make_unknown_arity_apply_widget env ap returns exn a))
  ∧ pv (fst (make_unknown_arity_apply_widget env ap returns exn a)).
Proof.
  intros Hp. unfold make_unknown_arity_apply_widget, fresh.
  cbv beta iota.
  match goal with
  | |- context [fresh_list ?n ?b] =>
    destruct (AW.fresh_list_spec n b) as (_ & _ & _ & _ & _ & Hq);
    destruct (fresh_list n b) as [ws a2] eqn:E2
  end.
  po_let E2 po_fresh_list. cbn [fst] in Hq.
  match goal with
  | |- context [unknown_arity_apply_deps ?r ?x ?c ?l ?b] =>
    assert (H3 : po_ok b (unknown_arity_apply_deps r x c l b));
    [ | destruct ws as [ | w ws']; cbn [fst snd]; split; try exact I ]
  end.
  - apply po_apply_deps.
    + intros x [<- | [<- | Hx]]; [ apply Hp; left; reflexivity | exact I | ].
      apply Hp. right. exact Hx.
    + intros first w H. apply in_combine_r in H.
      destruct (Hq w H) as (m & -> & _). exact I.
  - eapply po_trans; [ | apply po_same; reflexivity ].
    eapply po_trans; [ | exact H3 ].
    eapply po_trans; [ | apply po_note_simples ].
    eapply po_trans; [ | exact Hpo ].
    unfold add_fact. apply po_add_facts_at; try reflexivity. po_free.
  - eapply po_trans; [ | apply po_cond_alias ].
    + eapply po_trans; [ | apply po_same; reflexivity ].
      eapply po_trans; [ | exact H3 ].
      eapply po_trans; [ | apply po_note_simples ].
      eapply po_trans; [ | exact Hpo ].
      unfold add_fact. apply po_add_facts_at; try reflexivity. po_free.
    + destruct (Hq w (or_introl eq_refl)) as (m & -> & _). exact I.
Qed.

Lemma po_add_apply (env : traverse_env) (id : code_id) (callee widget : node)
    (only : bool) (a : acc) :
  pv widget -> po_ok a (add_apply env id callee widget only a).
Proof.
  intros Hw. unfold add_apply. destruct only; [ | apply po_push_apply_dep;
    exact Hw ].
  unfold fresh. cbv beta iota.
  eapply po_trans; [ | apply po_push_apply_dep; exact I ].
  apply po_add_facts_at; try reflexivity. po_free.
Qed.

Lemma po_call_kind (env : traverse_env) (ap : apply_expr) (exn_arg : node)
    (return_args : list node) (a : acc) :
  (∀ x, x ∈ exn_arg :: return_args -> pv x) ->
  po_ok a (traverse_call_kind env ap exn_arg return_args a).
Proof.
  intros Hp. unfold traverse_call_kind.
  apply (po_trans _ _ _ (po_note_simple (ap_callee ap) a)).
  destruct (ap_call_kind ap) as [id | | code_ids].
  - destruct (po_make_known env ap return_args exn_arg
                (note_simple (ap_callee ap) a) Hp) as [H4 Hcw].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg
                (note_simple (ap_callee ap) a)) as [cw a4] eqn:E4.
    cbn [fst snd] in *. apply (po_trans _ _ _ H4).
    match goal with
    | |- context [add_fact ?f a4] =>
      apply (po_trans _ (add_fact f a4)); [ apply po_add_fact; po_node | ]
    end.
    destruct (env_should_preserve_direct_calls env);
      try (apply po_add_apply; exact Hcw).
    destruct (negb _); [ apply po_add_apply; exact Hcw | apply po_refl ].
  - destruct (po_make_unknown env ap return_args exn_arg
                (note_simple (ap_callee ap) a) Hp) as [H4 Hcw].
    destruct (make_unknown_arity_apply_widget env ap return_args exn_arg
                (note_simple (ap_callee ap) a)) as [cw a4] eqn:E4.
    cbn [fst snd] in *. apply (po_trans _ _ _ H4).
    apply po_add_fact. po_node.
  - destruct (po_make_known env ap return_args exn_arg
                (note_simple (ap_callee ap) a) Hp) as [H4 Hcw].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg
                (note_simple (ap_callee ap) a)) as [cw a4] eqn:E4.
    cbn [fst snd] in *. apply (po_trans _ _ _ H4).
    apply po_add_fact. po_node.
Qed.

Lemma po_traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  penv env -> po_ok a (traverse_apply env ap a).
Proof.
  intros Henv. unfold traverse_apply.
  destruct (po_find_cont env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a Henv) as [H1 Hr].
  destruct (find_cont env (ap_continuation ap) _ a) as [ra a1] eqn:E1.
  destruct (po_find_cont env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) a1 Henv) as [H2 He].
  destruct (find_cont env (ap_exn_continuation ap) _ a1) as [ep a2] eqn:E2.
  cbn [fst snd] in *.
  apply (po_trans _ _ _ H1), (po_trans _ _ _ H2).
  destruct ep as [ | p ps].
  - destruct (fresh a2) as [x a3] eqn:E3. po_let E3 po_fresh.
    apply (po_trans _ _ _ Hpo). apply po_call_kind.
    intros y [<- | Hy]; [ | exact (Hr y Hy) ].
    unfold fresh in E3. injection E3 as <- _. exact I.
  - cbv beta iota.
    eapply po_trans; [ | apply po_call_kind ].
    + eapply po_trans; [ apply po_note_simples | ].
      apply po_add_facts. po_free. apply pv_node_ok, He. right.
      eapply in_combine_l. eassumption.
    + intros x [<- | Hx]; [ apply He; left; reflexivity | exact (Hr x Hx) ].
Qed.

Lemma po_apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc) :
  penv env -> po_ok a (apply_cont_deps env ac a).
Proof.
  intros Henv. unfold apply_cont_deps.
  destruct (po_find_cont env (ac_continuation ac) (length (ac_args ac)) a
              Henv) as [H1 Hp].
  destruct (find_cont env (ac_continuation ac) _ a) as [ps a1] eqn:E1.
  cbn [fst snd] in *. apply (po_trans _ _ _ H1).
  apply (po_trans _ _ _ (po_note_simples (ac_args ac) a1)).
  apply po_add_facts. po_free. apply pv_node_ok, Hp.
  eapply in_combine_l. eassumption.
Qed.

Lemma po_traverse_switch (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  penv env -> po_ok a (traverse_switch env scrutinee arms a).
Proof.
  intros Henv. unfold traverse_switch.
  apply (po_trans _ _ _ (po_cond_any_usage env scrutinee a)).
  apply po_fold. intros a' [i ac] _. apply po_apply_cont_deps. exact Henv.
Qed.

Lemma po_exn_handler_deps (env : traverse_env) (params : list variable)
    (md : handler_metadata) (body : expr) (a : acc) :
  (∀ p, p ∈ params -> p ∈ param_variables e0) ->
  po_ok a (exn_handler_deps env (Handler params md body) a).
Proof.
  intros Hp. unfold exn_handler_deps. destruct params as [ | p ps];
    [ apply po_refl | ].
  destruct (hm_is_exn_handler md); [ | apply po_refl ].
  eapply po_trans; [ | apply po_cond_any_usage ].
  apply po_add_fact. po_node. left. apply Hp. left. reflexivity.
Qed.

Lemma po_code_aliases (my_closure : variable) (params : list variable)
    (cd : code_dep) (a : acc) :
  my_closure ∈ param_variables e0 ->
  (∀ p, p ∈ params -> p ∈ param_variables e0) ->
  po_ok a
    (add_facts
       (Fact_alias (Node_variable my_closure) (cd_my_closure cd)
        :: map (fun '(p, arg) => Fact_alias (Node_variable p) arg)
             (combine params (cd_params cd)))
       a).
Proof.
  intros Hm Hp. apply po_add_facts. po_free.
  - left. exact Hm.
  - left. apply Hp. eapply in_combine_l. eassumption.
Qed.

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

(* What a node held at the function slot [g] is: made; the variable of a
   [Let] of a closure; that of a projection of [g] off the variable of a
   [Let] that holds it at [g]; or a symbol of a static closure. *)
Definition held_ok (a : acc) (g : function_slot) (b : node) : Prop :=
  (∃ k, b = Node_made k)
  ∨ (∃ y slot decls vs, b = Node_variable y
       ∧ (y, Closure_of_set slot decls vs) ∈ let_bindings e0)
  ∨ (∃ y s z, b = Node_variable y
       ∧ (y, Project_function_slot s g (Var z)) ∈ let_bindings e0
       ∧ Fact_constructor (Node_variable z) (Field_function_slot g) b
         ∈ acc_facts a)
  ∨ (∃ sym t slot cs decls vs, b = Node_symbol sym
       ∧ (t, Static_const_closure_of_set slot cs decls vs)
         ∈ CE.sym_bindings e0
       ∧ (g, sym) ∈ compact cs).

Lemma held_mono (a a' : acc) (g : function_slot) (b : node) :
  (∀ fct, fct ∈ acc_facts a -> fct ∈ acc_facts a') ->
  held_ok a g b -> held_ok a' g b.
Proof.
  intros Hs [H | [H | [(y & s & z & H1 & H2 & H3) | H]]];
    [ left | right; left | right; right; left | right; right; right ];
    auto.
  exists y, s, z. auto.
Qed.

Record jinv (a : acc) : Prop := {
  j_ok : ∀ fct, fct ∈ acc_facts a -> fact_ok fct;
  j_tgt : ∀ fct x, fct ∈ acc_facts a -> x ∈ tgt fct ->
    x ∉ cm a ∧ AW.made_lt x (acc_next a);
  j_cm : ∀ x, x ∈ cm a -> AW.made_lt x (acc_next a);
  j_cd : ∀ id cd x, (id, cd) ∈ acc_code_deps a -> x ∈ AW.code_dep_nodes cd ->
    x ∉ cm a ∧ AW.made_lt x (acc_next a) ∧ pv x;
  j_ad : ∀ d, d ∈ acc_apply_deps a ->
    ad_apply_call_witness d ∉ cm a
    ∧ AW.made_lt (ad_apply_call_witness d) (acc_next a)
    ∧ pv (ad_apply_call_witness d);
  j_held : ∀ x g b,
    Fact_constructor x (Field_function_slot g) b ∈ acc_facts a ->
    held_ok a g b
}.

(* The parameters of the continuations in scope are made before, held at
   no function slot, and of [pv]. *)
Record jenv (env : traverse_env) (a : acc) : Prop := {
  je_conts : ∀ k ps x, (k, ps) ∈ conts env -> x ∈ ps ->
    x ∉ cm a ∧ AW.made_lt x (acc_next a) ∧ pv x;
  je_sets : env_set_closures env = set_closures e0
}.

(* The nodes held at a function slot of [a'] are those of [a] and nodes
   made from [a]. *)
Definition grows (a a' : acc) : Prop :=
  (acc_next a <= acc_next a')%positive
  ∧ ∀ x, x ∈ cm a' ->
      x ∈ cm a ∨ ∃ k, x = Node_made k ∧ (acc_next a <= k)%positive.

Lemma not_cm_grows (a a' : acc) (x : node) :
  grows a a' -> x ∉ cm a -> AW.made_lt x (acc_next a) ->
  x ∉ cm a' ∧ AW.made_lt x (acc_next a').
Proof.
  intros [Hn Hc] Hx Hlt. split.
  - intros H. destruct (Hc x H) as [H' | (k & -> & Hk)]; [ exact (Hx H') | ].
    specialize (Hlt k eq_refl). lia.
  - intros k E. specialize (Hlt k E). lia.
Qed.

Lemma grows_refl (a : acc) : grows a a.
Proof. split; [ lia | intros x H; left; exact H ]. Qed.

Lemma grows_trans (a a1 a2 : acc) : grows a a1 -> grows a1 a2 -> grows a a2.
Proof.
  intros [H1 H2] [H3 H4]. split; [ lia | ].
  intros x Hx. destruct (H4 x Hx) as [H | (k & -> & Hk)].
  - destruct (H2 x H) as [H' | (k & -> & Hk)]; [ left; exact H' | ].
    right. exists k. split; [ reflexivity | lia ].
  - right. exists k. split; [ reflexivity | lia ].
Qed.

Lemma jenv_grows (env : traverse_env) (a a' : acc) :
  grows a a' -> jenv env a -> jenv env a'.
Proof.
  intros Hg [Hc Hs]. constructor; [ | exact Hs ].
  intros k ps x Hk Hx. destruct (Hc k ps x Hk Hx) as (H1 & H2 & H3).
  destruct (not_cm_grows a a' x Hg H1 H2). auto.
Qed.

Definition jstep (env : traverse_env) (a a' : acc) : Prop :=
  jinv a -> jenv env a -> jinv a' ∧ grows a a'.

Lemma jstep_refl (env : traverse_env) (a : acc) : jstep env a a.
Proof. intros H _. split; [ exact H | apply grows_refl ]. Qed.

Lemma jstep_trans (env : traverse_env) (a a1 a2 : acc) :
  jstep env a a1 -> jstep env a1 a2 -> jstep env a a2.
Proof.
  intros H1 H2 Hinv Henv. destruct (H1 Hinv Henv) as [Hinv1 Hg1].
  destruct (H2 Hinv1 (jenv_grows _ _ _ Hg1 Henv)) as [Hinv2 Hg2].
  split; [ exact Hinv2 | exact (grows_trans _ _ _ Hg1 Hg2) ].
Qed.

Lemma jstep_fold {A : Type} (env : traverse_env) (f : acc -> A -> acc)
    (l : list A) (a : acc) :
  (∀ y b, y ∈ l -> jstep env b (f b y)) -> jstep env a (List.fold_left f l a).
Proof.
  revert a. induction l as [ | y l IH]; intros a H; cbn [List.fold_left].
  - apply jstep_refl.
  - apply (jstep_trans _ _ (f a y)); [ apply H; left; reflexivity | ].
    apply IH. intros y' b Hy'. apply H. right. exact Hy'.
Qed.

Lemma jstep_env (env env' : traverse_env) (a a' : acc) :
  (jinv a -> jenv env a -> jenv env' a) -> jstep env' a a' -> jstep env a a'.
Proof. intros H H' Hinv Henv. exact (H' Hinv (H Hinv Henv)). Qed.

(* ======================================================================== *)
(* The steps that add no slot fact                                          *)
(* ======================================================================== *)

(* What the steps of [KnownWitnesses] and [AutoWidgets] give from the
   ghost accumulators: what is put into or made is no node held at a
   function slot. *)
Definition mk_ok (a a' : acc) : Prop :=
  (acc_next a <= acc_next a')%positive
  ∧ (∃ L, acc_facts a' = L ++ acc_facts a
        ∧ ∀ fct x, fct ∈ L -> x ∈ AW.into_nodes fct ->
            x ∉ cm a ∧ AW.made_lt x (acc_next a'))
  ∧ (∀ id cd, (id, cd) ∈ acc_code_deps a' ->
       (id, cd) ∈ acc_code_deps a
       ∨ ∀ x, x ∈ AW.code_dep_nodes cd ->
           x ∉ cm a ∧ AW.made_lt x (acc_next a'))
  ∧ (∀ d, d ∈ acc_apply_deps a' ->
       d ∈ acc_apply_deps a
       ∨ (ad_apply_call_witness d ∉ cm a
          ∧ AW.made_lt (ad_apply_call_witness d) (acc_next a'))).

Lemma cm_lt (a : acc) :
  jinv a -> ∀ y, y ∈ cm a -> ∃ k, y = Node_made k ∧ (k < acc_next a)%positive.
Proof.
  intros Hinv y Hy. destruct (cm_made a y Hy) as [k ->]. exists k.
  split; [ reflexivity | exact (j_cm a Hinv _ Hy k eq_refl) ].
Qed.

Lemma mk_of_kw (a a' : acc) :
  jinv a -> KW.ext (UC.ghost (acc_next a) (cm a)) a a' -> mk_ok a a'.
Proof.
  intros Hinv [Hn (L & Hf & Hi & _) Hc Hap _].
  pose proof (fun x => UC.ok_ghost (acc_next a) (cm a) a' x (cm_lt a Hinv) Hn)
    as Hok.
  split; [ exact Hn | split; [ | split ] ].
  - exists L. split; [ exact Hf | ]. intros fct x H Hx.
    destruct (Hok x (Hi fct x H Hx)) as (H1 & H2 & _). auto.
  - intros id cd H. left. rewrite <- Hc. exact H.
  - intros d H. destruct (Hap d H) as [H' | H']; [ left; exact H' | right ].
    destruct (Hok _ H') as (H1 & H2 & _). auto.
Qed.

(* An accumulator whose widgets are the nodes [sp], for the steps of
   [AutoWidgets]. *)
Definition ghost_w (n : positive) (sp : list node) : acc :=
  {| acc_next := n;
     acc_facts :=
       map (fun x => Fact_accessor x
                       (Field_call_witness Known_arity_code_pointer) x) sp;
     acc_code_deps := []; acc_apply_deps := []; acc_closure_deps := [];
     acc_symbols := [] |}.

Lemma ok_ghost_w (n : positive) (sp : list node) (a : acc) (x : node) :
  (∀ y, y ∈ sp -> ∃ k, y = Node_made k ∧ (k < n)%positive) ->
  (n <= acc_next a)%positive ->
  AW.ok_node (ghost_w n sp) a x -> x ∉ sp ∧ AW.made_lt x (acc_next a).
Proof.
  intros Hsp Hn [H | [[H1 H2] | (k & -> & H1 & H2)]].
  - split; [ | intros k E; destruct (H k E) ].
    intros Hx. destruct (Hsp x Hx) as (k & -> & _). exact (H k eq_refl).
  - split; [ | intros k E; specialize (H1 k E); cbn in H1; lia ].
    intros Hx. apply H2. exists x. cbn. apply in_map_iff. exists x. auto.
  - cbn [ghost_w acc_next] in H1. split.
    + intros Hx. destruct (Hsp _ Hx) as (k' & [= <-] & Hk). lia.
    + intros k' [= <-]. exact H2.
Qed.

Lemma mk_of_aw (a a' : acc) :
  jinv a -> AW.ext (ghost_w (acc_next a) (cm a)) a a' -> mk_ok a a'.
Proof.
  intros Hinv [Hn (L & Hf & Hi & _) Hc Hap _].
  pose proof (fun x => ok_ghost_w (acc_next a) (cm a) a' x (cm_lt a Hinv) Hn)
    as Hok.
  split; [ exact Hn | split; [ | split ] ].
  - exists L. split; [ exact Hf | ]. intros fct x H Hx.
    exact (Hok x (Hi fct x H Hx)).
  - intros id cd H. destruct (Hc id cd H) as [H' | H']; [ left; exact H' | ].
    right. intros x Hx. exact (Hok x (H' x Hx)).
  - intros d H. destruct (Hap d H) as [H' | H']; [ left; exact H' | right ].
    exact (Hok _ H').
Qed.

(* A step that adds no slot fact, and facts, deps and nodes of [mk_ok] and
   [po_ok]. *)
Lemma jcore (a a' : acc) :
  jinv a -> mk_ok a a' -> CE.ns_ok a a' -> po_ok a a' ->
  jinv a' ∧ grows a a'.
Proof.
  intros Hinv (Hn & (L & Hf & Hi) & Hc & Hap) Hns (P1 & P2 & P3).
  assert (Hcm : ∀ x, x ∈ cm a' <-> x ∈ cm a)
    by (intros x; exact (cm_ns a a' x Hns)).
  assert (Hsub : ∀ fct, fct ∈ acc_facts a -> fct ∈ acc_facts a')
    by (destruct Hns as (H & _); exact H).
  assert (Hold : ∀ x, x ∉ cm a -> AW.made_lt x (acc_next a) ->
             x ∉ cm a' ∧ AW.made_lt x (acc_next a')).
  { intros x H1 H2. split; [ rewrite Hcm; exact H1 | ].
    exact (AW.made_lt_mono _ _ _ H2 Hn). }
  split; [ constructor
         | split; [ exact Hn | intros x Hx; left; apply Hcm; exact Hx ] ].
  - intros fct Hin. destruct (P1 fct Hin) as [H | H];
      [ exact (j_ok a Hinv fct H) | exact H ].
  - intros fct x Hin Hx. rewrite Hf in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + destruct (Hi fct x Hin (tgt_into fct x Hx)) as [H1 H2].
      split; [ rewrite Hcm; exact H1 | exact H2 ].
    + destruct (j_tgt a Hinv fct x Hin Hx) as [H1 H2]. exact (Hold x H1 H2).
  - intros x Hx. apply Hcm in Hx.
    exact (AW.made_lt_mono _ _ _ (j_cm a Hinv x Hx) Hn).
  - intros id cd x Hin Hx.
    destruct (P2 id cd Hin) as [Hin' | Hpv].
    + destruct (j_cd a Hinv id cd x Hin' Hx) as (H1 & H2 & H3).
      destruct (Hold x H1 H2). auto.
    + destruct (Hc id cd Hin) as [Hin' | H].
      * destruct (j_cd a Hinv id cd x Hin' Hx) as (H1 & H2 & H3).
        destruct (Hold x H1 H2). auto.
      * destruct (H x Hx) as [H1 H2].
        split; [ rewrite Hcm; exact H1 | ].
        split; [ exact H2 | exact (Hpv x Hx) ].
  - intros d Hd. destruct (P3 d Hd) as [Hd' | Hpv].
    + destruct (j_ad a Hinv d Hd') as (H1 & H2 & H3).
      destruct (Hold _ H1 H2). auto.
    + destruct (Hap d Hd) as [Hd' | [H1 H2]].
      * destruct (j_ad a Hinv d Hd') as (H1 & H2 & H3).
        destruct (Hold _ H1 H2). auto.
      * split; [ rewrite Hcm; exact H1 | split; [ exact H2 | exact Hpv ] ].
  - intros x g b Hin. apply (held_mono a a'); [ exact Hsub | ].
    apply (j_held a Hinv x). destruct Hns as (_ & H2 & _).
    apply H2; [ exact Hin | cbn; reflexivity ].
Qed.

Lemma jenv_ghost (env : traverse_env) (a : acc) :
  jenv env a -> KW.env_good env (UC.ghost (acc_next a) (cm a)).
Proof.
  intros Henv k ps x Hk Hx.
  destruct (je_conts env a Henv k ps x Hk Hx) as (H1 & H2 & _).
  split; [ exact H2 | intros Hw; exact (H1 (proj1 (UC.kwit_ghost _ _ _) Hw)) ].
Qed.

Lemma jenv_penv (env : traverse_env) (a : acc) : jenv env a -> penv env.
Proof.
  intros Henv k ps x Hk Hx. exact (proj2 (proj2 (je_conts env a Henv k ps x
                                                    Hk Hx))).
Qed.

Lemma jstep_leaf (env : traverse_env) (a a' : acc) :
  (jinv a -> jenv env a ->
   KW.ext (UC.ghost (acc_next a) (cm a)) a a' ∧ CE.ns_ok a a' ∧ po_ok a a') ->
  jstep env a a'.
Proof.
  intros H Hinv Henv. destruct (H Hinv Henv) as (H1 & H2 & H3).
  exact (jcore a a' Hinv (mk_of_kw a a' Hinv H1) H2 H3).
Qed.

Lemma ghost_le (a : acc) :
  (acc_next (UC.ghost (acc_next a) (cm a)) <= acc_next a)%positive.
Proof. cbn. lia. Qed.

Lemma jstep_prepare_binding (env : traverse_env) (b : static_binding)
    (a : acc) :
  jstep env a (prepare_binding b a).
Proof.
  intros Hinv _. apply (jcore a); [ exact Hinv | | apply CE.ns_prepare_binding
                                   | apply po_prepare_binding ].
  apply mk_of_aw; [ exact Hinv | ]. apply AW.ext_prepare_binding. cbn. lia.
Qed.

(* ======================================================================== *)
(* The steps that add slot facts                                            *)
(* ======================================================================== *)

Lemma jstep_slots (a a' : acc) (L : list fact) :
  jinv a -> (acc_next a <= acc_next a')%positive ->
  acc_facts a' = L ++ acc_facts a ->
  (∀ fct, fct ∈ L -> tgt fct = []) ->
  (∀ x g b, Fact_constructor x (Field_function_slot g) b ∈ L ->
     held_ok a' g b) ->
  (∀ y, y ∈ flat_map cm_of L ->
     ∃ k, y = Node_made k ∧ (acc_next a <= k)%positive
          ∧ (k < acc_next a')%positive) ->
  acc_code_deps a' = acc_code_deps a -> acc_apply_deps a' = acc_apply_deps a ->
  jinv a' ∧ grows a a'.
Proof.
  intros Hinv Hn HL Ht Hh Hm Hc Hd.
  assert (Hg : grows a a').
  { split; [ exact Hn | ]. intros x Hx.
    destruct (cm_app L a a' x HL Hx) as [H | H]; [ right | left; exact H ].
    destruct (Hm x H) as (k & -> & Hk & _). eauto. }
  assert (Hsub : ∀ fct, fct ∈ acc_facts a -> fct ∈ acc_facts a')
    by (intros fct H; rewrite HL; apply in_or_app; right; exact H).
  split; [ constructor | exact Hg ].
  - intros fct Hin. rewrite HL in Hin. apply in_app_or in Hin as [Hin | Hin].
    + intros x Hx. rewrite (Ht fct Hin) in Hx. destruct Hx.
    + exact (j_ok a Hinv fct Hin).
  - intros fct x Hin Hx. rewrite HL in Hin.
    apply in_app_or in Hin as [Hin | Hin];
      [ rewrite (Ht fct Hin) in Hx; destruct Hx | ].
    destruct (j_tgt a Hinv fct x Hin Hx) as [H1 H2].
    exact (not_cm_grows a a' x Hg H1 H2).
  - intros x Hx. destruct (cm_app L a a' x HL Hx) as [H | H].
    + destruct (Hm x H) as (k & -> & _ & Hk). intros k' [= <-]. exact Hk.
    + exact (AW.made_lt_mono _ _ _ (j_cm a Hinv x H) Hn).
  - intros id cd x Hin Hx. rewrite Hc in Hin.
    destruct (j_cd a Hinv id cd x Hin Hx) as (H1 & H2 & H3).
    destruct (not_cm_grows a a' x Hg H1 H2). auto.
  - intros d' Hd'. rewrite Hd in Hd'.
    destruct (j_ad a Hinv d' Hd') as (H1 & H2 & H3).
    destruct (not_cm_grows a a' _ Hg H1 H2). auto.
  - intros x g b Hin. rewrite HL in Hin.
    apply in_app_or in Hin as [Hin | Hin]; [ exact (Hh x g b Hin) | ].
    exact (held_mono a a' g b Hsub (j_held a Hinv x g b Hin)).
Qed.

Lemma record_set_cm (env : traverse_env)
    (closures : list (node * option code_id))
    (fs : list (function_slot * node)) (vs : list (value_slot * simple))
    (L : list fact) (y : node) :
  (∀ fct : fact,
     fct ∈ L
     <-> ∃ (base : node) (code : option code_id),
           (base, code) ∈ closures
           ∧ ((∃ (w : value_slot) (s : simple),
                 (w, s)
                 ∈ filter (fun '(w, _) => should_keep_value_slot env w)
                     (compact vs)
                 ∧ fct
                   = Fact_constructor base (Field_value_slot w)
                       (node_of_simple s))
              ∨ ∃ (f : function_slot) (c : node),
                  (f, c) ∈ fs
                  ∧ fct = Fact_constructor base (Field_function_slot f) c)) ->
  y ∈ flat_map cm_of L -> ∃ f, (f, y) ∈ fs.
Proof.
  intros HLf Hy. apply in_flat_map in Hy as (fct & Hin & Hy).
  destruct (cm_of_in fct y Hy) as (y0 & g & k & -> & ->).
  apply HLf in Hin
    as (base & code & _ & [(w & s & _ & E) | (f & c & Hfc & E)]);
    [ discriminate E | ].
  injection E as _ _ <-. eauto.
Qed.

Lemma record_set_tgt (env : traverse_env)
    (closures : list (node * option code_id))
    (fs : list (function_slot * node)) (vs : list (value_slot * simple))
    (L : list fact) :
  (∀ fct : fact,
     fct ∈ L
     <-> ∃ (base : node) (code : option code_id),
           (base, code) ∈ closures
           ∧ ((∃ (w : value_slot) (s : simple),
                 (w, s)
                 ∈ filter (fun '(w, _) => should_keep_value_slot env w)
                     (compact vs)
                 ∧ fct
                   = Fact_constructor base (Field_value_slot w)
                       (node_of_simple s))
              ∨ ∃ (f : function_slot) (c : node),
                  (f, c) ∈ fs
                  ∧ fct = Fact_constructor base (Field_function_slot f) c)) ->
  ∀ fct, fct ∈ L -> tgt fct = [].
Proof.
  intros HLf fct Hin.
  apply HLf in Hin
    as (base & code & _ & [(w & s & _ & ->) | (f & c & _ & ->)]);
    reflexivity.
Qed.

Lemma jstep_record_plain (env : traverse_env)
    (closures : list (node * option code_id))
    (vs : list (value_slot * simple)) (a : acc) :
  jstep env a (record_set_of_closures_deps env closures [] vs a).
Proof.
  intros Hinv _.
  destruct (CE.record_set_facts env closures [] vs a) as [(L & HL & HLf) Hn].
  destruct (record_set_deps env closures [] vs a) as [Hc Hd].
  apply (jstep_slots a _ L Hinv); try assumption.
  - rewrite Hn. lia.
  - exact (record_set_tgt env closures [] vs L HLf).
  - intros x g b Hin. apply HLf in Hin
      as (base & code & _ & [(w & s & _ & E) | (f & c & [] & _)]).
    discriminate E.
  - intros y Hy. destruct (record_set_cm env closures [] vs L y HLf Hy)
      as (f & []).
Qed.

Lemma jstep_static_closure (env : traverse_env) (s : symbol)
    (slot : function_slot) (cs : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (a : acc) :
  (s, Static_const_closure_of_set slot cs decls vs) ∈ CE.sym_bindings e0 ->
  jstep env a
    (traverse_static_binding env
       (Bind_symbol s (Static_const_closure_of_set slot cs decls vs)) a).
Proof.
  intros Hs Hinv _. cbn [traverse_static_binding].
  match goal with
  | |- context [record_set_of_closures_deps env ?cl ?fs vs a] =>
    destruct (CE.record_set_facts env cl fs vs a) as [(L & HL & HLf) Hn];
    destruct (record_set_deps env cl fs vs a) as [Hc Hd];
    pose proof (record_set_tgt env cl fs vs L HLf) as Ht;
    pose proof (record_set_cm env cl fs vs L) as Hm
  end.
  apply (jstep_slots a _ L Hinv); try assumption.
  - rewrite Hn. lia.
  - intros x g b Hin. apply HLf in Hin
      as (base & code & _ & [(w & s0 & _ & E) | (f & c & Hfc & E)]);
      [ discriminate E | ].
    injection E as _ <- <-.
    apply in_map_iff in Hfc as ([f' sym] & Heq & Hin). injection Heq as <- <-.
    right; right; right. exists sym, s, slot, cs, decls, vs. auto.
  - intros y Hy. destruct (Hm y HLf Hy) as (f & Hf).
    apply in_map_iff in Hf as ([f' sym] & Heq & _). injection Heq as _ <-.
    apply in_flat_map in Hy as (fct & _ & Hy).
    destruct (cm_of_in fct _ Hy) as (y0 & g & k & _ & E). discriminate E.
Qed.

Lemma jstep_closure_let (env : traverse_env) (x : variable)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (a : acc) :
  (x, Closure_of_set slot decls vs) ∈ let_bindings e0 ->
  jstep env a (traverse_closure_of_set env x slot decls vs a).
Proof.
  intros Hx Hinv Henv. unfold traverse_closure_of_set.
  destruct (among slot (map fst (compact decls))) eqn:Hslot;
    [ | split; [ exact Hinv | apply grows_refl ] ].
  destruct (closures_of_set_deps x slot (set_closures_of env x)
              (compact decls) a) as (D1 & D2 & D3).
  destruct (closures_of_set x slot (set_closures_of env x) (compact decls) a)
    as [cl a1] eqn:Ecl.
  destruct (CE.closures_of_set_spec x slot _ _ a cl a1 Ecl)
    as (_ & Hn1 & Hmap & Hcl).
  cbn [snd] in D1, D2, D3.
  set (closures := map (fun '(_, name, id) => (name, Some id)) cl).
  set (fs := map (fun '(f, name, _) => (f, name)) cl).
  destruct (CE.record_set_facts env closures fs vs a1) as [(L & HL & HLf) Hn2].
  destruct (record_set_deps env closures fs vs a1) as [Hc Hd].
  assert (Hxin : ∃ id0, (Node_variable x, Some id0) ∈ closures).
  { apply among_In in Hslot.
    assert (Hslot' : slot ∈ map (fun '(f, _, _) => f) cl)
      by (rewrite Hmap; exact Hslot).
    apply in_map_iff in Hslot' as ([[f1 N] id] & Heq & Hin1). cbn in Heq.
    subst f1. destruct (Hcl slot N id Hin1) as (Hs & _).
    rewrite (Hs eq_refl) in Hin1.
    exists id. unfold closures. apply in_map_iff.
    exists (slot, Node_variable x, id). auto. }
  apply (jstep_slots a _ L Hinv).
  - rewrite Hn2. exact Hn1.
  - rewrite HL, D1. reflexivity.
  - exact (record_set_tgt env closures fs vs L HLf).
  - intros y g b Hin. pose proof Hin as Hin0. apply HLf in Hin0
      as (base & code & _ & [(w & s & _ & E) | (f & c & Hfc & E)]);
      [ discriminate E | ].
    injection E as _ -> ->.
    pose proof Hfc as Hfc0.
    apply in_map_iff in Hfc as ([[f' N] id] & Heq & Hcl').
    injection Heq as -> ->.
    destruct (Hcl f c id Hcl') as (_ & [-> | [(y' & Hy' & ->) | (k & -> & _)]]).
    + right; left. exists x, slot, decls, vs. auto.
    + right; right; left.
      unfold set_closures_of in Hy'. rewrite (je_sets env a Henv) in Hy'.
      destruct (lookup x (set_closures e0)) as [projs | ] eqn:Hp;
        [ | discriminate Hy' ].
      apply lookup_in in Hy'.
      destruct (proj1 CE.set_closures_projs e0 x projs (lookup_in _ _ _ Hp)
                  f y' Hy') as (s & Hs).
      exists y', s, x. split; [ reflexivity | split; [ exact Hs | ] ].
      rewrite HL. apply in_or_app. left. apply HLf.
      destruct Hxin as (id0 & Hid0).
      exists (Node_variable x), (Some id0). split; [ exact Hid0 | right ].
      exists f, (Node_variable y'). auto.
    + left. eauto.
  - intros y Hy. destruct (record_set_cm env closures fs vs L y HLf Hy)
      as (f & Hf).
    apply in_map_iff in Hf as ([[f' N] id] & Heq & Hcl').
    injection Heq as -> ->.
    apply in_flat_map in Hy as (fct & _ & Hy).
    destruct (cm_of_in fct _ Hy) as (y0 & g & k & _ & ->).
    destruct (Hcl f (Node_made k) id Hcl')
      as (_ & [E | [(y' & _ & E) | (k' & E & Hk1 & Hk2)]]);
      try discriminate E.
    injection E as <-. exists k. split; [ reflexivity | ].
    rewrite Hn2. auto.
  - rewrite Hc. exact D2.
  - rewrite Hd. exact D3.
Qed.

(* ======================================================================== *)
(* The traversal                                                            *)
(* ======================================================================== *)

(* A part of [e0]. *)
Definition sub (e : expr) : Prop :=
  (∀ p, p ∈ let_bindings e -> p ∈ let_bindings e0)
  ∧ (∀ v, v ∈ param_variables e -> v ∈ param_variables e0)
  ∧ (∀ p, p ∈ CE.sym_bindings e -> p ∈ CE.sym_bindings e0).

Definition P_j (e : expr) : Prop :=
  ∀ (env : traverse_env) (a : acc), sub e -> jstep env a (traverse env e a).

Lemma jenv_add_cont (env : traverse_env) (k : continuation)
    (params : list variable) (a : acc) :
  (∀ p, p ∈ params -> p ∈ param_variables e0) ->
  jinv a -> jenv env a -> jenv (add_cont env k (map Node_variable params)) a.
Proof.
  intros Hp Hinv Henv. constructor; [ | exact (je_sets env a Henv) ].
  intros k' ps x [Heq | Hin] Hx.
  - injection Heq as <- <-. apply in_map_iff in Hx as (p & <- & Hp').
    split; [ intros H; destruct (cm_made a _ H) as [m E]; discriminate E | ].
    split; [ intros m E; discriminate E | exact (Hp p Hp') ].
  - exact (je_conts env a Henv k' ps x Hin Hx).
Qed.

Lemma jenv_add_rec_conts (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) (a : acc) :
  (∀ k params md body, (k, Handler params md body) ∈ handlers ->
     ∀ p, p ∈ params -> p ∈ param_variables e0) ->
  jinv a -> jenv env a -> jenv (add_rec_conts env handlers) a.
Proof.
  intros Hp Hinv Henv. unfold add_rec_conts.
  induction handlers as [ | [k [params md body]] hs IH]; cbn [fold_right];
    [ exact Henv | ].
  apply jenv_add_cont; [ | exact Hinv | ].
  - exact (Hp k params md body (or_introl eq_refl)).
  - apply IH. intros k' ps md' b' H. exact (Hp k' ps md' b' (or_intror H)).
Qed.

Lemma jenv_code_env (env : traverse_env) (id : code_id)
    (ret exn : continuation) (cd : code_dep) (a : acc) :
  (∃ id', (id', cd) ∈ acc_code_deps a) -> jinv a -> jenv env a ->
  jenv (code_env env id ret exn cd) a.
Proof.
  intros (id' & Hcd) Hinv Henv. constructor; [ | exact (je_sets env a Henv) ].
  intros k ps x Hk Hx. cbn [code_env FlambdaReaperTraversal.env_conts] in Hk.
  destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
    apply (j_cd a Hinv id' cd x Hcd); cbn [AW.code_dep_nodes].
  - right; left. destruct Hx as [<- | []]. reflexivity.
  - right; right. exact Hx.
Qed.

Lemma jstep_traverse :
  (∀ e : expr, P_j e)
  ∧ ∀ h : continuation_handler,
      match h with Handler _ _ handler => P_j handler end.
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind; unfold P_j.
  - intros x k d body IH env a (HL & HP & HS).
    cbn [let_bindings param_variables CE.sym_bindings] in HL, HP, HS.
    cbn [traverse].
    assert (Hx : (x, d) ∈ let_bindings e0) by (apply HL; left; reflexivity).
    apply (jstep_trans _ _ (traverse_named env x d a));
      [ | apply IH; split; [ intros p Hp; apply HL; right; exact Hp
                           | split; assumption ] ].
    destruct d as [s | p args | id vs | slot decls vs | w c | pf w c
                  | mf mt c];
      try (apply jstep_leaf; intros Hinv Henv;
           split; [ apply KW.ext_traverse_named, ghost_le | ];
           split; [ apply CE.ns_traverse_named; intros; discriminate
                  | apply po_traverse_named; exact Hx ]).
    + cbn [traverse_named]. apply jstep_record_plain.
    + cbn [traverse_named]. apply jstep_closure_let. exact Hx.
  - intros bindings body Hcodes IH env a (HL & HP & HS).
    cbn [let_bindings param_variables CE.sym_bindings] in HL, HP, HS.
    cbn [traverse].
    eapply jstep_trans;
      [ | apply IH; split;
          [ intros p Hp; apply HL; apply in_or_app; right; exact Hp
          | split; intros p Hp; [ apply HP | apply HS ]; apply in_or_app;
            right; exact Hp ] ].
    eapply jstep_trans; [ | apply jstep_fold ].
    + eapply jstep_trans; [ | apply jstep_fold ].
      * apply jstep_fold. intros b a' _. apply jstep_prepare_binding.
      * intros b a' Hb.
        assert (Hsb : ∀ s c, b = Bind_symbol s c ->
                        (s, c) ∈ CE.sym_bindings e0).
        { intros s c ->. apply HS. apply in_or_app. left.
          apply in_flat_map. exists (Bind_symbol s c).
          split; [ exact Hb | left; reflexivity ]. }
        cbv beta.
        destruct b as [id c | id | s const];
          [ apply jstep_refl | apply jstep_refl | ].
        destruct const as [id vs | slot cs decls vs | | | | ];
          try (apply jstep_leaf; intros Hinv Henv;
               split; [ apply KW.ext_traverse_static_binding | ];
               split; [ apply CE.ns_traverse_static_binding; intros;
                        discriminate
                      | apply po_traverse_static_binding; exact Hsb ]).
        -- cbn [traverse_static_binding]. apply jstep_record_plain.
        -- apply jstep_static_closure. exact (Hsb _ _ eq_refl).
    + intros [id c | id | s const] a' Hb; cbv beta; cbn [traverse_binding];
        [ | apply jstep_refl | apply jstep_refl ].
      destruct c as [params mc ret exn md cbody]. cbn [traverse_code].
      destruct (lookup id (acc_code_deps a')) as [cd | ] eqn:Ecd;
        [ | apply jstep_refl ].
      assert (Hin : ∀ (B : Type) (g : static_binding -> list B) (y : B),
                 y ∈ g (Bind_code id (Code params mc ret exn md cbody)) ->
                 y ∈ flat_map g bindings).
      { intros B g y Hy. apply in_flat_map.
        exists (Bind_code id (Code params mc ret exn md cbody)). auto. }
      assert (Hpc : ∀ v, v ∈ mc :: params ++ param_variables cbody ->
                      v ∈ param_variables e0).
      { intros v Hv. apply HP. apply in_or_app. left.
        exact (Hin _ binding_param_variables v Hv). }
      eapply jstep_trans.
      * apply jstep_leaf. intros Hinv Henv.
        split; [ | split; [ apply CE.ns_code_aliases
                          | apply po_code_aliases ] ].
        -- apply KW.ext_add_facts.
           ++ intros fct y H Hy. apply KW.ok_not_made.
              destruct H as [<- | H];
                [ | apply in_map_iff in H as ([p arg] & <- & _) ];
                cbn in Hy; destruct Hy as [<- | []]; intros k E;
                discriminate E.
           ++ intros b from [H | H]; [ discriminate H | ].
              apply in_map_iff in H as ([p arg] & H & _). discriminate H.
        -- apply Hpc. left. reflexivity.
        -- intros p Hp. apply Hpc. right. apply in_or_app. left. exact Hp.
      * apply (jstep_env _ (code_env env id ret exn cd)).
        -- intros Hinv Henv. apply jenv_code_env; [ | exact Hinv | exact Henv ].
           exists id. exact (lookup_in _ _ _ Ecd).
        -- pose proof (Hcodes id (Code params mc ret exn md cbody) Hb) as Hc.
           cbn in Hc. apply Hc. split; [ | split ].
           ++ intros p Hp. apply HL. apply in_or_app. left.
              exact (Hin _ binding_let_bindings p Hp).
           ++ intros v Hv. apply Hpc. right. apply in_or_app. right.
              exact Hv.
           ++ intros p Hp. apply HS. apply in_or_app. left.
              exact (Hin _ CE.binding_sym_bindings p Hp).
  - intros k h body IHh IH env a (HL & HP & HS).
    destruct h as [params md handler].
    cbn [let_bindings param_variables CE.sym_bindings handler_let_bindings
         handler_param_variables CE.handler_sym_bindings] in HL, HP, HS.
    cbn [traverse traverse_handler] in IHh |- *.
    assert (Hpp : ∀ p, p ∈ params -> p ∈ param_variables e0).
    { intros p Hp. apply HP. apply in_or_app. left. apply in_or_app. left.
      exact Hp. }
    eapply jstep_trans; [ apply IHh; split; [ | split ] | ].
    + intros p Hp. apply HL. apply in_or_app. left. exact Hp.
    + intros v Hv. apply HP. apply in_or_app. left. apply in_or_app. right.
      exact Hv.
    + intros p Hp. apply HS. apply in_or_app. left. exact Hp.
    + eapply jstep_trans.
      * apply jstep_leaf. intros Hinv Henv.
        split; [ apply KW.ext_exn_handler_deps | ].
        split; [ apply CE.ns_exn_handler_deps | ].
        apply po_exn_handler_deps. exact Hpp.
      * apply (jstep_env _ (add_cont env k (map Node_variable params))).
        -- intros Hinv Henv. apply jenv_add_cont; assumption.
        -- apply IH. split; [ | split ]; intros ? H;
             [ apply HL | apply HP | apply HS ]; apply in_or_app; right;
             exact H.
  - intros handlers body IHh IH env a (HL & HP & HS).
    cbn [let_bindings param_variables CE.sym_bindings] in HL, HP, HS.
    cbn [traverse].
    assert (Hin : ∀ (B : Type) (g : continuation_handler -> list B) k h y,
               (k, h) ∈ handlers -> y ∈ g h ->
               y ∈ flat_map (fun '(_, h) => g h) handlers).
    { intros B g k h y Hk Hy. apply in_flat_map. exists (k, h). auto. }
    assert (Hhp : ∀ k params md hb, (k, Handler params md hb) ∈ handlers ->
                    ∀ p, p ∈ params -> p ∈ param_variables e0).
    { intros k params md hb Hk p Hp. apply HP. apply in_or_app. left.
      apply (Hin _ handler_param_variables k _ p Hk). cbn.
      apply in_or_app. left. exact Hp. }
    apply (jstep_env _ (add_rec_conts env handlers));
      [ intros Hinv Henv; apply jenv_add_rec_conts; assumption | ].
    eapply jstep_trans;
      [ | apply IH; split; [ | split ]; intros ? H;
          [ apply HL | apply HP | apply HS ]; apply in_or_app; right;
          exact H ].
    apply jstep_fold. intros [k h] a' Hkh. destruct h as [params md hb].
    pose proof (IHh k _ Hkh) as Hh. cbn [traverse_handler] in Hh |- *.
    apply Hh. split; [ | split ].
    + intros p Hp. apply HL. apply in_or_app. left.
      exact (Hin _ handler_let_bindings k _ p Hkh Hp).
    + intros v Hv. apply HP. apply in_or_app. left.
      apply (Hin _ handler_param_variables k _ v Hkh). cbn.
      apply in_or_app. right. exact Hv.
    + intros p Hp. apply HS. apply in_or_app. left.
      exact (Hin _ CE.handler_sym_bindings k _ p Hkh Hp).
  - intros ap env a _. apply jstep_leaf. intros Hinv Henv.
    split; [ apply KW.ext_traverse_apply;
             [ exact (jenv_ghost env a Henv) | apply ghost_le ] | ].
    split; [ apply CE.ns_traverse_apply | ].
    apply po_traverse_apply. exact (jenv_penv env a Henv).
  - intros ac env a _. apply jstep_leaf. intros Hinv Henv.
    split; [ apply KW.ext_apply_cont_deps;
             [ exact (jenv_ghost env a Henv) | apply ghost_le ] | ].
    split; [ apply CE.ns_apply_cont_deps | ].
    apply po_apply_cont_deps. exact (jenv_penv env a Henv).
  - intros scrutinee arms env a _. apply jstep_leaf. intros Hinv Henv.
    split; [ apply KW.ext_traverse_switch;
             [ exact (jenv_ghost env a Henv) | apply ghost_le ] | ].
    split; [ apply CE.ns_traverse_switch | ].
    apply po_traverse_switch. exact (jenv_penv env a Henv).
  - intros env a _. apply jstep_refl.
  - intros params md handler IH. exact IH.
Qed.

(* ======================================================================== *)
(* The delayed deps and the unit                                            *)
(* ======================================================================== *)

Lemma tgt_alias_for_caller (caller : option code_id) (from to_ : node) :
  tgt (alias_for_caller caller from to_) = [to_].
Proof. destruct caller; reflexivity. Qed.

Lemma closure_deps_ok (a : acc) :
  jinv a ->
  ∀ (ds : list closure_dep) (b : acc), (acc_next a <= acc_next b)%positive ->
    ∃ L, acc_facts (List.fold_left
                      (fun b d => add_closure_dep (acc_code_deps a) d b) ds b)
         = L ++ acc_facts b
      ∧ ∀ fct, fct ∈ L -> fact_ok fct ∧ ∀ x, x ∈ tgt fct -> x ∉ cm a.
Proof.
  intros Hinv ds. induction ds as [ | d ds IH]; intros b Hb;
    cbn [List.fold_left].
  - exists []. split; [ reflexivity | intros fct [] ].
  - assert (H1 : ∃ L1,
               acc_facts (add_closure_dep (acc_code_deps a) d b)
               = L1 ++ acc_facts b
               ∧ (acc_next b
                  <= acc_next (add_closure_dep (acc_code_deps a) d b))%positive
               ∧ ∀ fct, fct ∈ L1 ->
                   fact_ok fct ∧ ∀ x, x ∈ tgt fct -> x ∉ cm a).
    { unfold add_closure_dep.
      destruct (lookup (cld_closure_code_id d) (acc_code_deps a))
        as [cd | ] eqn:E.
      - eexists. split; [ reflexivity | split; [ cbn; lia | ] ].
        destruct (j_cd a Hinv _ cd (cd_my_closure cd) (lookup_in _ _ _ E)
                    (or_introl eq_refl)) as (Hm1 & _ & Hm3).
        intros fct Hin.
        assert (Hin' : tgt fct = [cd_my_closure cd] ∨ tgt fct = []).
        { destruct (cd_unknown_arity_call_witnesses cd); cbn in Hin;
            repeat destruct Hin as [<- | Hin]; cbn; auto; destruct Hin. }
        unfold fact_ok. destruct Hin' as [Ht | Ht]; rewrite Ht;
          [ | split; intros x [] ].
        split; intros x [<- | []]; [ apply pv_node_ok; exact Hm3 | exact Hm1 ].
      - unfold fresh. cbv beta iota.
        eexists. split; [ reflexivity | split; [ cbn; lia | ] ].
        intros fct Hin.
        assert (Hin' : tgt fct = [Node_made (acc_next b)] ∨ tgt fct = []).
        { cbn in Hin. repeat destruct Hin as [<- | Hin]; cbn; auto.
          destruct Hin. }
        unfold fact_ok. destruct Hin' as [Ht | Ht]; rewrite Ht;
          [ | split; intros x [] ].
        split; intros x [<- | []]; [ exact I | ].
        intros Hc. specialize (j_cm a Hinv _ Hc _ eq_refl). lia. }
    destruct H1 as (L1 & HL1 & Hn1 & HF1).
    destruct (IH (add_closure_dep (acc_code_deps a) d b)) as (L2 & HL2 & HF2);
      [ lia | ].
    exists (L2 ++ L1). split; [ rewrite HL2, HL1, app_assoc; reflexivity | ].
    intros fct Hin. apply in_app_or in Hin as [Hin | Hin];
      [ exact (HF2 fct Hin) | exact (HF1 fct Hin) ].
Qed.

Lemma resolve_ok (a : acc) :
  jinv a ->
  let a' := resolve_delayed_deps (static_names symbols_of_binding e0) a in
  (∀ fct x, fct ∈ acc_facts a' -> x ∈ tgt fct -> node_ok fct x ∧ x ∉ cm a')
  ∧ ∀ x g b, Fact_constructor x (Field_function_slot g) b ∈ acc_facts a' ->
      held_ok a' g b.
Proof.
  intros Hinv a'.
  pose proof (CE.ns_resolve (static_names symbols_of_binding e0) a) as Hns.
  assert (Hcm : ∀ y, y ∈ cm a' <-> y ∈ cm a)
    by (intros y; exact (cm_ns a a' y Hns)).
  assert (Hnew : ∀ fct, fct ∈ acc_facts a' ->
             fct ∈ acc_facts a
             ∨ (fact_ok fct ∧ ∀ x, x ∈ tgt fct -> x ∉ cm a)).
  { unfold a', resolve_delayed_deps. cbv zeta.
    set (b1 := add_facts _ a).
    assert (Hb1 : ∀ fct, fct ∈ acc_facts b1 ->
                fct ∈ acc_facts a
                ∨ (fact_ok fct ∧ ∀ x, x ∈ tgt fct -> x ∉ cm a)).
    { intros fct Hin. unfold b1 in Hin. cbn [add_facts acc_facts] in Hin.
      apply in_app_or in Hin as [Hin | Hin]; [ right | left; exact Hin ].
      apply in_flat_map in Hin as (d & Hd & Hin).
      destruct (j_ad a Hinv d Hd) as (Hw1 & _ & Hw3).
      unfold apply_dep_facts in Hin.
      destruct (lookup (ad_apply_code_id d) (acc_code_deps a))
        as [cd | ] eqn:E.
      - destruct (j_cd a Hinv _ cd (cd_my_closure cd) (lookup_in _ _ _ E)
                    (or_introl eq_refl)) as (Hm1 & _ & Hm3).
        unfold fact_ok.
        destruct Hin as [<- | [<- | []]]; rewrite tgt_alias_for_caller;
          split; intros x [<- | []];
          try (apply pv_node_ok; assumption); assumption.
      - unfold fact_ok.
        destruct (ad_function_containing_apply_expr d);
          destruct Hin as [<- | [<- | []]]; cbn [tgt];
          split; intros x Hx; cbn [In] in Hx; try contradiction;
          destruct Hx as [<- | []]; try exact I;
          try (apply pv_node_ok; assumption); try assumption;
          intros Hc; destruct (cm_made a _ Hc) as [m Em]; discriminate Em. }
    destruct (closure_deps_ok a Hinv (acc_closure_deps b1) b1)
      as (L2 & HL2 & HF2); [ cbn; lia | ].
    intros fct Hin. cbn [add_facts acc_facts] in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    - right. apply in_map_iff in Hin as (s & <- & Hs).
      unfold without in Hs. apply filter_In in Hs as [_ Hs].
      apply negb_true_iff in Hs.
      split; intros x [<- | []].
      + right. intros Hin. apply among_In in Hin. congruence.
      + intros Hc. destruct (cm_made a _ Hc) as [m Em]. discriminate Em.
    - rewrite HL2 in Hin. apply in_app_or in Hin as [Hin | Hin];
        [ right; exact (HF2 fct Hin) | exact (Hb1 fct Hin) ]. }
  split.
  - intros fct x Hin Hx. destruct (Hnew fct Hin) as [H | [H1 H2]].
    + split; [ exact (j_ok a Hinv fct H x Hx) | ].
      rewrite Hcm. exact (proj1 (j_tgt a Hinv fct x H Hx)).
    + split; [ exact (H1 x Hx) | rewrite Hcm; exact (H2 x Hx) ].
  - intros x g b Hin. destruct Hns as (Hsub & H2 & _).
    apply (held_mono a a' g b Hsub). apply (j_held a Hinv x).
    apply H2; [ exact Hin | reflexivity ].
Qed.

Lemma unit_final (u : unit_interface) :
  ∃ af : acc,
    traverse_unit Auto u e0 = facts_of_list (acc_facts af)
    ∧ (∀ fct x, fct ∈ acc_facts af -> x ∈ tgt fct ->
         node_ok fct x ∧ x ∉ cm af)
    ∧ ∀ x g b, Fact_constructor x (Field_function_slot g) b ∈ acc_facts af ->
        held_ok af g b.
Proof.
  unfold traverse_unit.
  set (a0 := add_fact (Fact_any_source Node_all_constants) empty_acc).
  destruct (fresh a0) as [r a1] eqn:E1.
  destruct (fresh a1) as [x a2] eqn:E2.
  cbv beta iota zeta.
  unfold fresh in E1, E2. injection E1 as <- <-. injection E2 as <- <-.
  match goal with
  | |- context [traverse ?env e0 ?b] => set (env0 := env); set (a3 := b)
  end.
  assert (Hcm3 : cm a3 = []) by reflexivity.
  assert (Hinv3 : jinv a3).
  { constructor; rewrite ?Hcm3; cbn;
      try (intros; contradiction).
    - intros fct [<- | [<- | []]]; po_node.
    - intros fct y [<- | [<- | []]] Hy; cbn in Hy; try contradiction;
        destruct Hy as [<- | []]; split; [ intros [] | intros k E;
                                           discriminate E ].
    - intros y g b [E | [E | []]]; discriminate E. }
  assert (Henv3 : jenv env0 a3).
  { constructor; [ | reflexivity ].
    intros k ps y Hk Hy. rewrite Hcm3. cbn in Hk.
    destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
      destruct Hy as [<- | []];
      (split; [ intros [] | split; [ intros m [= <-]; cbn; lia | exact I ] ]).
  }
  assert (Hsub0 : sub e0) by (split; [ auto | split; auto ]).
  destruct (proj1 jstep_traverse e0 env0 a3 Hsub0 Hinv3 Henv3)
    as [Hinv4 _].
  set (a4 := traverse env0 e0 a3) in *.
  assert (Hinv5 : jinv (add_fact (Fact_any_usage (Node_made 1)) a4)).
  { apply (jcore a4); [ exact Hinv4 | | apply CE.ns_add_fact; cbn; auto
                      | apply po_add_fact; po_node ].
    apply mk_of_kw; [ exact Hinv4 | ]. apply KW.ext_add_fact.
    - intros y [].
    - intros b from E. discriminate E. }
  eexists. split; [ reflexivity | ].
  exact (resolve_ok _ Hinv5).
Qed.

End Facts.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Lemma nodup_app_not (l1 l2 : list positive) (x : positive) :
  NoDup (l1 ++ l2) -> x ∈ l1 -> x ∈ l2 -> False.
Proof.
  induction l1 as [ | y l1 IH]; intros Hnd H1 H2; [ destruct H1 | ].
  cbn [app] in Hnd. inversion Hnd as [ | y' l' Hy Hnd' ]; subst.
  destruct H1 as [<- | H1]; [ | exact (IH Hnd' H1 H2) ].
  apply Hy. apply in_or_app. right. exact H2.
Qed.

Section Final.

Variable e : expr.
Hypothesis Hnl : NoDup (map fst (let_bindings e)).
Hypothesis Hns : NoDup (map fst (CE.sym_bindings e)).
Hypothesis Hwf : sets_well_formed e = true.
Hypothesis Hold : CE.no_old_sets e.
Hypothesis Hb : NoDup (map fst (let_bindings e) ++ param_variables e).

(* What the facts of a unit put into a node held at a function slot is the
   read of its slot off a variable that holds it there. *)
Lemma into_closure (af : acc) :
  (∀ fct x, fct ∈ acc_facts af -> x ∈ tgt fct ->
     node_ok e fct x ∧ x ∉ cm af) ->
  (∀ x g b, Fact_constructor x (Field_function_slot g) b ∈ acc_facts af ->
     held_ok e af g b) ->
  ∀ fct x g b,
    Fact_constructor x (Field_function_slot g) b ∈ acc_facts af ->
    fct ∈ acc_facts af -> b ∈ tgt fct ->
    ∃ h z, fct = Fact_accessor b (Field_function_slot h) (Node_variable z)
           ∧ Fact_constructor (Node_variable z) (Field_function_slot h) b
             ∈ acc_facts af.
Proof.
  intros Hok Hheld fct x g b Hc Hin Hbt.
  destruct (Hok fct b Hin Hbt) as [Hn Hcm].
  assert (Hlet : ∀ y d, (y, d) ∈ let_bindings e -> y ∈ param_variables e ->
                   False).
  { intros y d Hy Hp. apply (nodup_app_not _ _ y Hb); [ | exact Hp ].
    exact (in_map fst _ (y, d) Hy). }
  destruct (Hheld x g b Hc)
    as [(k & ->)
       | [(y & slot & decls & vs & -> & Hy)
         | [(y & s & z & -> & Hy & Hz)
           | (sym & t & slot & cs & decls & vs & -> & Ht & Hgs)]]].
  - exfalso. apply Hcm. apply cm_in. eauto.
  - exfalso. cbn [node_ok] in Hn. destruct Hn as [Hp | (d & Hd & Hnt)];
      [ exact (Hlet y _ Hy Hp) | ].
    pose proof (CE.nodup_same _ _ _ _ Hnl Hy Hd) as E. subst d. exact Hnt.
  - cbn [node_ok] in Hn. destruct Hn as [Hp | (d & Hd & Hnt)];
      [ destruct (Hlet y _ Hy Hp) | ].
    pose proof (CE.nodup_same _ _ _ _ Hnl Hy Hd) as E. subst d.
    cbn in Hnt. exists g, z. split; [ exact Hnt | exact Hz ].
  - exfalso. apply in_compact, lookup_in in Hgs.
    pose proof (proj2 (proj1 CE.static_sets_whole e Hwf t slot cs decls vs Ht)
                  g sym Hgs) as Hsym.
    cbn [node_ok] in Hn. destruct Hn as [(c & Hsc & Hcl) | Hnot].
    + pose proof (CE.nodup_same _ _ _ _ Hns Hsym Hsc) as E. subst c.
      discriminate Hcl.
    + apply Hnot. rewrite <- (proj1 CE.sym_bindings_static_names e Hold).
      exact (in_map fst _ _ Hsym).
Qed.

End Final.

Theorem closures_unaliased_traversal :
  closures_unaliased_traversal_statement.
Proof.
  intros u e Hfrag Hwfu Honce.
  destruct (TT.unit_premises u e Hfrag Hwfu Honce)
    as (Hold & Hnl & Hns & Hwf).
  assert (Hb : NoDup (map fst (let_bindings e) ++ param_variables e)).
  { unfold variables_bound_once in Honce. apply distinct_NoDup in Honce.
    exact Honce. }
  destruct (unit_final e u) as (af & Ha & Hok & Hheld).
  pose proof (into_closure e Hnl Hns Hwf Hold Hb af Hok Hheld) as Hinto.
  rewrite Ha. intros x b g Hc. apply CE.constructor_of_list in Hc.
  assert (Hno : ∀ fct, fct ∈ acc_facts af -> b ∈ tgt fct ->
             (∀ h z, fct <> Fact_accessor b (Field_function_slot h)
                              (Node_variable z)) -> False).
  { intros fct Hin Hbt Hne.
    destruct (Hinto fct x g b Hc Hin Hbt) as (h & z & E & _).
    exact (Hne h z E). }
  assert (Hfl : ∀ (B : Type) (gf : fact -> list B) (y : B),
             y ∈ flat_map gf (acc_facts af) ->
             ∃ fct, fct ∈ acc_facts af ∧ y ∈ gf fct)
    by (intros B gf y H; apply in_flat_map in H; exact H).
  split; [ | split; [ | split; [ | split; [ | split;
    [ | split; [ | split; [ | split ] ] ] ] ] ] ].
  - intros from H.
    apply FlambdaReaperWrapperAutoProofs.alias_in in H.
    apply (Hno _ H); [ left; reflexivity | intros h z E; discriminate E ].
  - intros c from H.
    apply FlambdaReaperWrapperAutoProofs.propagate_in in H.
    apply (Hno _ H); [ left; reflexivity | intros h z E; discriminate E ].
  - intros c from H.
    apply FlambdaReaperWrapperAutoProofs.if_any_source_in in H.
    apply (Hno _ H); [ left; reflexivity | intros h z E; discriminate E ].
  - intros base i H.
    apply FlambdaReaperWrapperAutoProofs.parameter_in in H.
    apply (Hno _ H); [ left; reflexivity | intros h z E; discriminate E ].
  - intros H. apply Hfl in H as (fct & Hin & Hy).
    destruct fct; cbn in Hy; try contradiction.
    destruct Hy as [<- | []].
    apply (Hno _ Hin); [ left; reflexivity | intros h z E; discriminate E ].
  - intros [].
  - intros c H. apply Hfl in H as (fct & Hin & Hy).
    destruct fct; cbn in Hy; try contradiction.
    destruct Hy as [Hy | []]. injection Hy as -> ->.
    apply (Hno _ Hin); [ left; reflexivity | intros h z E; discriminate E ].
  - intros from H. apply Hfl in H as (fct & Hin & Hy).
    destruct fct; cbn in Hy; try contradiction.
    destruct Hy as [Hy | []]. injection Hy as -> ->.
    apply (Hno _ Hin); [ left; reflexivity | intros h z E; discriminate E ].
  - intros f base H.
    apply FlambdaReaperWrapperAutoProofs.accessor_in in H.
    destruct (Hinto _ x g b Hc H (or_introl eq_refl)) as (h & z & E & Hz).
    injection E as -> ->. exists h. split; [ reflexivity | ].
    apply CE.constructor_of_list. exact Hz.
Qed.

Theorem changed_type_sound_full : changed_type_sound_full_statement.
Proof.
  intros is_local u T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' fuel m t n v v_mid v' Hfrag Hwfu Honce
    Hleast Hlocal Hsets Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc Huses
    Hunst Ht Hcis Hrv Hat Hch Hsat.
  exact (TT.changed_type_sound_traversal is_local u T sets dominated
           to_unbox tcr e tenv canon statics statics' valuation valuation'
           fuel m t n v v_mid v' Hfrag Hwfu Honce Hleast Hlocal Hsets Henv
           Hfragt Halloc Hst Hwhole Hcopies Hdomc Huses Hunst
           (closures_unaliased_traversal u e Hfrag Hwfu Honce)
           Ht Hcis Hrv Hat Hch Hsat).
Qed.
