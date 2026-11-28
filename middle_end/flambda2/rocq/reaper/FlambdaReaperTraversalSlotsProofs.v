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

(* The proofs of FlambdaReaperTraversalSlotsSpec.v. Nothing here needs
   review.

   [inv]: each fact of the accumulator puts something into the variable
   of a projection of a value slot only if it is its read ([var_ok]), and
   reads a value slot only as a projection does ([slot_read_ok]); the
   nodes of the deps that facts are later put into ([free]: the nodes of
   the code deps, the call witnesses of the apply deps and the closures
   of the closure deps) are not such variables. The steps of the
   traversal keep it: the [Let] of a projection adds its read; every
   other step puts things into parameters, into the variables of other
   [Let]s, into symbols and into made nodes, all [free] where the unit
   binds each variable once. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperCallingConventions FlambdaReaperLeastSpec
  FlambdaReaperTraversalSlotsSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs FlambdaReaperLeastProofs
  FlambdaReaperClosuresExactProofs
  FlambdaReaperIndirectDeadFramesTraversalProofs.
Import ListNotations.

Module AW := FlambdaReaperLeastProofs.AutoWidgets.
Module CE := FlambdaReaperClosuresExactProofs.
Module CH := FlambdaReaperIndirectDeadFramesTraversalProofs.Chains.

Abbreviation into_nodes := AW.into_nodes.
Abbreviation conts := FlambdaReaperTraversal.env_conts.

Section Slots.

Variable e0 : expr.

(* ======================================================================== *)
(* What a fact may put into a variable                                      *)
(* ======================================================================== *)

(* Facts may put anything into [x]: it is not the variable of a
   projection of a value slot. *)
Definition free (x : node) : Prop :=
  ∀ (v : variable) (d : named),
    x = Node_variable v -> (v, d) ∈ let_bindings e0 ->
    slot_projection d = None.

(* What [fct] may put into the variable [v]: anything, unless [v] is that
   of a projection of a value slot, and then only its read. *)
Definition var_ok (fct : fact) (v : variable) : Prop :=
  ∀ (d : named) (w : value_slot) (c : simple),
    (v, d) ∈ let_bindings e0 -> slot_projection d = Some (w, c) ->
    fct
    = Fact_accessor (Node_variable v) (Field_value_slot w) (node_of_simple c).

(* A read of a value slot is that of a projection. *)
Definition slot_read_ok (fct : fact) : Prop :=
  match fct with
  | Fact_accessor to_ (Field_value_slot w) b =>
    ∃ (v : variable) (d : named) (c : simple),
      to_ = Node_variable v ∧ (v, d) ∈ let_bindings e0
      ∧ slot_projection d = Some (w, c) ∧ b = node_of_simple c
  | _ => True
  end.

Definition fact_ok (fct : fact) : Prop :=
  (∀ v : variable, Node_variable v ∈ into_nodes fct -> var_ok fct v)
  ∧ slot_read_ok fct.

Lemma free_var_ok (fct : fact) (v : variable) :
  free (Node_variable v) -> var_ok fct v.
Proof.
  intros H d w c Hd Hs. rewrite (H v d eq_refl Hd) in Hs. discriminate Hs.
Qed.

Lemma free_made (k : positive) : free (Node_made k).
Proof. intros v d E. discriminate E. Qed.

Lemma free_symbol (s : symbol) : free (Node_symbol s).
Proof. intros v d E. discriminate E. Qed.

Lemma free_code_id (id : code_id) : free (Node_code_id id).
Proof. intros v d E. discriminate E. Qed.

Lemma free_consts : free Node_all_constants.
Proof. intros v d E. discriminate E. Qed.

Lemma ok_free (fct : fact) :
  (∀ x, x ∈ into_nodes fct -> free x) -> slot_read_ok fct -> fact_ok fct.
Proof.
  intros H1 H2. split; [ | exact H2 ]. intros v Hv. apply free_var_ok.
  exact (H1 _ Hv).
Qed.

(* The nodes of the deps that facts are put into are [free]. *)
Definition inv (a : acc) : Prop :=
  (∀ fct, fct ∈ acc_facts a -> fact_ok fct)
  ∧ (∀ id cd, (id, cd) ∈ acc_code_deps a ->
       ∀ x, x ∈ AW.code_dep_nodes cd -> free x)
  ∧ (∀ d, d ∈ acc_apply_deps a -> free (ad_apply_call_witness d))
  ∧ (∀ d, d ∈ acc_closure_deps a ->
       free (cld_let_bound_name_of_the_closure d)).

(* A node of [free]. *)
Ltac free_tac :=
  first
    [ apply free_made | apply free_symbol | apply free_code_id
    | apply free_consts | assumption
    | match goal with
      | H : ∀ y : node, y ∈ _ -> free y |- free _ =>
        apply H;
        first
          [ left; reflexivity | right; left; reflexivity
          | eapply AW.indexed_in; eassumption
          | right; eapply AW.indexed_in; eassumption
          | right; right; eapply AW.indexed_in; eassumption
          | eapply in_combine_l; eassumption
          | eapply in_combine_r; eassumption
          | eassumption ]
      end ].

(* The facts of a list are of [fact_ok], their nodes [free]. *)
Ltac ok_list :=
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
  (apply ok_free;
   [ let x := fresh "x" in let Hx := fresh "Hx" in
     intros x Hx; cbn [AW.into_nodes In] in Hx;
     repeat match type of Hx with _ ∨ _ => destruct Hx as [Hx | Hx] end;
     try contradiction; subst x; try free_tac
   | cbn [slot_read_ok]; try exact I ]).

(* ======================================================================== *)
(* Steps that keep [inv]                                                    *)
(* ======================================================================== *)

Lemma inv_same (a a' : acc) :
  acc_facts a' = acc_facts a -> acc_code_deps a' = acc_code_deps a ->
  acc_apply_deps a' = acc_apply_deps a ->
  acc_closure_deps a' = acc_closure_deps a -> inv a -> inv a'.
Proof.
  intros E1 E2 E3 E4 H. unfold inv. rewrite E1, E2, E3, E4. exact H.
Qed.

Lemma inv_add_facts (L : list fact) (a : acc) :
  (∀ fct, fct ∈ L -> fact_ok fct) -> inv a -> inv (add_facts L a).
Proof.
  intros HL (H1 & H2 & H3 & H4). split; [ | exact (conj H2 (conj H3 H4)) ].
  intros fct Hin. cbn [add_facts acc_facts] in Hin.
  apply in_app_or in Hin as [Hin | Hin];
    [ exact (HL _ Hin) | exact (H1 _ Hin) ].
Qed.

Lemma inv_add_fact (fct : fact) (a : acc) :
  fact_ok fct -> inv a -> inv (add_fact fct a).
Proof. intros H. apply inv_add_facts. intros f [<- | []]. exact H. Qed.

Lemma inv_fresh (a : acc) : inv a -> inv (snd (fresh a)).
Proof. apply inv_same; reflexivity. Qed.

Lemma inv_fresh_list (n : nat) (a : acc) :
  inv a -> inv (snd (fresh_list n a)).
Proof.
  destruct (AW.fresh_list_spec n a) as (H1 & H2 & H3 & H4 & _).
  apply inv_same; assumption.
Qed.

Lemma inv_note_simple (s : simple) (a : acc) : inv a -> inv (note_simple s a).
Proof. destruct s; apply inv_same; reflexivity. Qed.

Lemma inv_note_simples (l : list simple) (a : acc) :
  inv a -> inv (note_simples l a).
Proof.
  destruct (AW.note_simples_fields l a) as (_ & H1 & H2 & H3 & H4).
  apply inv_same; assumption.
Qed.

Lemma inv_push_apply_dep (d : apply_dep) (a : acc) :
  free (ad_apply_call_witness d) -> inv a -> inv (push_apply_dep d a).
Proof.
  intros Hd (H1 & H2 & H3 & H4).
  split; [ exact H1 | split; [ exact H2 | split; [ | exact H4 ] ] ].
  intros d' [<- | Hin]; [ exact Hd | exact (H3 _ Hin) ].
Qed.

Lemma inv_push_closure_dep (d : closure_dep) (a : acc) :
  free (cld_let_bound_name_of_the_closure d) -> inv a ->
  inv (push_closure_dep d a).
Proof.
  intros Hd (H1 & H2 & H3 & H4).
  split; [ exact H1 | split; [ exact H2 | split; [ exact H3 | ] ] ].
  intros d' [<- | Hin]; [ exact Hd | exact (H4 _ Hin) ].
Qed.

Lemma inv_add_code_dep (id : code_id) (cd : code_dep) (a : acc) :
  (∀ x, x ∈ AW.code_dep_nodes cd -> free x) -> inv a ->
  inv (add_code_dep id cd a).
Proof.
  intros Hc (H1 & H2 & H3 & H4).
  split; [ exact H1 | split; [ | exact (conj H3 H4) ] ].
  intros id' cd' [Heq | Hin];
    [ injection Heq as <- <-; exact Hc | exact (H2 _ _ Hin) ].
Qed.

Lemma inv_fold {A : Type} (f : acc -> A -> acc) (l : list A) :
  (∀ (a : acc) (y : A), y ∈ l -> inv a -> inv (f a y)) ->
  ∀ a : acc, inv a -> inv (List.fold_left f l a).
Proof.
  induction l as [ | y l IH]; intros Hf a Ha; [ exact Ha | ].
  cbn [List.fold_left]. apply IH.
  - intros a' y' Hy'. apply Hf. right. exact Hy'.
  - apply Hf; [ left; reflexivity | exact Ha ].
Qed.

(* The parameters of the continuations in scope are [free]. *)
Definition penv (env : traverse_env) : Prop :=
  ∀ k ps x, (k, ps) ∈ conts env -> x ∈ ps -> free x.

Lemma inv_find_cont (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  penv env -> inv a ->
  inv (snd (find_cont env k n a))
  ∧ ∀ x, x ∈ fst (find_cont env k n a) -> free x.
Proof.
  intros Henv Ha. unfold find_cont.
  destruct (lookup k (conts env)) as [ps | ] eqn:E.
  - split; [ exact Ha | ]. intros x Hx.
    exact (Henv k ps x (lookup_in _ _ _ E) Hx).
  - destruct (AW.fresh_list_spec n a) as (_ & _ & _ & _ & _ & Hm).
    pose proof (inv_fresh_list n a Ha) as Hb.
    destruct (fresh_list n a) as [ps b] eqn:Eb. cbn [fst snd] in *.
    split.
    + apply inv_add_facts; [ | exact Hb ]. ok_list.
    + intros x Hx. destruct (Hm x Hx) as (m & -> & _). apply free_made.
Qed.

Lemma inv_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  inv a -> inv (cond_any_usage env s a).
Proof.
  intros Ha. unfold cond_any_usage. cbv zeta.
  apply (inv_note_simple s) in Ha.
  destruct (env_current_code_id env); apply inv_add_facts; try exact Ha;
    ok_list.
Qed.

Lemma inv_cond_alias (env : traverse_env) (from to_ : node) (a : acc) :
  free to_ -> inv a -> inv (cond_alias env from to_ a).
Proof.
  intros Ht Ha. unfold cond_alias.
  destruct (env_current_code_id env); apply inv_add_facts; try exact Ha;
    ok_list.
Qed.

Lemma inv_block_deps (base : node) (fields : list simple) (a : acc) :
  free base -> inv a -> inv (block_deps base fields a).
Proof.
  intros Hb Ha. unfold block_deps. cbv zeta.
  apply inv_add_facts; [ | apply inv_note_simples; exact Ha ]. ok_list.
Qed.

Lemma inv_record_set (env : traverse_env)
    (closures : list (node * option code_id))
    (fs : list (function_slot * node)) (vs : list (value_slot * simple))
    (a : acc) :
  (∀ base code, (base, code) ∈ closures -> free base) -> inv a ->
  inv (record_set_of_closures_deps env closures fs vs a).
Proof.
  unfold record_set_of_closures_deps. cbv zeta.
  set (vsk := filter (fun '(w, _) => should_keep_value_slot env w)
                (compact vs)).
  revert a. induction closures as [ | [base code] closures IH];
    intros a Hb Ha; [ exact Ha | ].
  cbn [List.fold_left]. apply IH.
  - intros b c H. exact (Hb b c (or_intror H)).
  - cbv beta iota.
    assert (Hbase : free base) by exact (Hb base code (or_introl eq_refl)).
    apply inv_add_facts; [ ok_list | ].
    apply inv_note_simples.
    destruct code; [ apply inv_push_closure_dep; [ exact Hbase | ] | ];
      exact Ha.
Qed.

Lemma inv_closures_of_set (x : variable) (slot : function_slot)
    (others : list (function_slot * variable)) :
  ∀ (decls : list (function_slot * code_id)) (a : acc),
    inv a -> inv (snd (closures_of_set x slot others decls a)).
Proof.
  induction decls as [ | [f id] decls IH]; intros a Ha; [ exact Ha | ].
  cbn [closures_of_set].
  destruct (if Pos.eqb f slot then (Node_variable x, a)
            else match lookup f others with
                 | Some y => (Node_variable y, a)
                 | None => fresh a
                 end) as [name a1] eqn:E1.
  assert (Ha1 : inv a1).
  { destruct (Pos.eqb f slot); [ injection E1 as _ <-; exact Ha | ].
    destruct (lookup f others); [ injection E1 as _ <-; exact Ha | ].
    change a1 with (snd (name, a1)). rewrite <- E1. apply inv_fresh.
    exact Ha. }
  specialize (IH a1 Ha1).
  destruct (closures_of_set x slot others decls a1) as [rest a2]. exact IH.
Qed.

Section Bound_once.

Hypothesis Hnl : NoDup (map fst (let_bindings e0)).
Hypothesis Hlp :
  ∀ v, v ∈ map fst (let_bindings e0) -> v ∈ param_variables e0 -> False.

Lemma free_param (v : variable) :
  v ∈ param_variables e0 -> free (Node_variable v).
Proof.
  intros Hp v' d E Hd. injection E as <-. exfalso.
  apply (Hlp v); [ | exact Hp ]. apply in_map_iff. exists (v, d). auto.
Qed.

Lemma free_let (v : variable) (d : named) :
  (v, d) ∈ let_bindings e0 -> slot_projection d = None ->
  free (Node_variable v).
Proof.
  intros Hd Hs v' d' E Hd'. injection E as <-.
  rewrite (CE.nodup_same _ _ _ _ Hnl Hd' Hd). exact Hs.
Qed.

Lemma inv_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  (x, Prim p args) ∈ let_bindings e0 -> inv a ->
  inv (traverse_prim env x p args a).
Proof.
  intros Hx Ha.
  assert (Hfx : free (Node_variable x)) by exact (free_let x _ Hx eq_refl).
  assert (Hacc : ∀ (f : field) (s : simple) (a0 : acc),
             (∀ w, f <> Field_value_slot w) -> inv a0 ->
             inv (add_fact (Fact_accessor (Node_variable x) f
                              (node_of_simple s)) (note_simple s a0))).
  { intros f s a0 Hf Ha0.
    apply inv_add_fact; [ | apply inv_note_simple; exact Ha0 ].
    apply ok_free; [ intros y [<- | []]; exact Hfx | ].
    destruct f; cbn [slot_read_ok]; try exact I.
    exfalso. exact (Hf _ eq_refl). }
  assert (Hdefault :
            inv (add_facts
                   (Fact_any_source (Node_variable x)
                    :: map (fun from => Fact_use (Node_variable x) from)
                         (free_names_of_simples args))
                   (if prim_has_arbitrary_effects p
                    then cond_any_usage env (Var x) a else a))).
  { apply inv_add_facts; [ ok_list | ].
    destruct (prim_has_arbitrary_effects p);
      [ apply inv_cond_any_usage | ]; exact Ha. }
  unfold traverse_prim; cbv zeta.
  destruct p as [u0 | b0 | v0 | h0 | r0];
    [ destruct u0 | | destruct v0 | destruct h0 | destruct r0 ];
    try (destruct args as [ | arg [ | arg2 args]]);
    cbv beta iota;
    try first [ exact Hdefault | apply inv_block_deps; assumption
              | exact Ha
              | apply Hacc; [ intros ? E; discriminate E | exact Ha ] ].
  all: try (match goal with b : bool |- _ => destruct b end;
            first [ exact Hdefault
                  | apply Hacc; [ intros ? E; discriminate E | exact Ha ] ]).
  - match goal with m : mutability |- _ => destruct m end;
      [ | apply Hacc; [ intros ? E; discriminate E | exact Ha ]
        | apply Hacc; [ intros ? E; discriminate E | exact Ha ] ].
    apply inv_add_fact;
      [ | apply Hacc; [ intros ? E; discriminate E | exact Ha ] ].
    apply ok_free; [ intros y [<- | []]; exact Hfx | exact I ].
  - apply inv_add_fact; [ | apply inv_note_simple; exact Ha ].
    apply ok_free; [ intros y [<- | []]; exact Hfx | exact I ].
Qed.

Lemma inv_traverse_named (env : traverse_env) (x : variable) (d : named)
    (a : acc) :
  env_set_closures env = set_closures e0 ->
  (x, d) ∈ let_bindings e0 -> inv a -> inv (traverse_named env x d a).
Proof.
  intros Hsc Hx Ha.
  assert (Hproj : ∀ (w : value_slot) (c : simple),
             slot_projection d = Some (w, c) ->
             fact_ok (Fact_accessor (Node_variable x) (Field_value_slot w)
                        (node_of_simple c))).
  { intros w c Hd. split.
    - intros v [Hv | []]. injection Hv as Hv. subst v.
      intros d' w' c' Hd' Hs'.
      rewrite (CE.nodup_same _ _ _ _ Hnl Hd' Hx), Hd in Hs'.
      injection Hs' as <- <-. reflexivity.
    - cbn [slot_read_ok]. exists x, d, c. auto. }
  destruct d as [s | p args | id vs | slot decls vs | w c | pf w c
                | mf mt c];
    cbn [traverse_named]; cbv zeta.
  - apply inv_add_fact; [ | apply inv_note_simple; exact Ha ].
    apply ok_free; [ | exact I ].
    intros y [<- | []]. exact (free_let x _ Hx eq_refl).
  - apply inv_traverse_prim; assumption.
  - apply inv_record_set; [ | exact Ha ].
    intros base code [Heq | []]. injection Heq as <- _.
    exact (free_let x _ Hx eq_refl).
  - unfold traverse_closure_of_set.
    destruct (among slot (map fst (compact decls))); [ | exact Ha ].
    pose proof (inv_closures_of_set x slot (set_closures_of env x)
                  (compact decls) a Ha) as Ha1.
    destruct (closures_of_set x slot (set_closures_of env x) (compact decls)
                a)
      as [cl a1] eqn:Ecl.
    destruct (CE.closures_of_set_spec x slot _ _ a cl a1 Ecl)
      as (_ & _ & _ & Hcl).
    cbn [snd] in Ha1.
    apply inv_record_set; [ | exact Ha1 ].
    intros base code Hin. apply in_map_iff in Hin as ([[f N] id] & Eq & Hin).
    injection Eq as <- _.
    destruct (proj2 (Hcl f N id Hin))
      as [-> | [(y & Hy & ->) | (k & -> & _)]].
    + exact (free_let x _ Hx eq_refl).
    + unfold set_closures_of in Hy. rewrite Hsc in Hy.
      destruct (lookup x (set_closures e0)) as [projs | ] eqn:Hp;
        [ | discriminate Hy ].
      destruct (proj1 CE.set_closures_projs e0 x projs (lookup_in _ _ _ Hp)
                  f y (lookup_in _ _ _ Hy))
        as (s & Hs).
      exact (free_let y _ Hs eq_refl).
    + apply free_made.
  - apply inv_add_fact; [ | apply inv_note_simple; exact Ha ].
    exact (Hproj w c eq_refl).
  - apply inv_add_fact; [ | apply inv_note_simple; exact Ha ].
    exact (Hproj w c eq_refl).
  - apply inv_add_fact; [ | apply inv_note_simple; exact Ha ].
    apply ok_free; [ | exact I ].
    intros y [<- | []]. exact (free_let x _ Hx eq_refl).
Qed.

End Bound_once.

Lemma inv_static_binding (env : traverse_env) (b : static_binding)
    (a : acc) :
  inv a -> inv (traverse_static_binding env b a).
Proof.
  intros Ha.
  destruct b as [id c | id | s const]; cbn [traverse_static_binding];
    try exact Ha.
  destruct const as [id vs | slot cs decls vs | tag fields | contents
                    | bytes | [ | el els]].
  - apply inv_record_set; [ | exact Ha ].
    intros base code [Heq | []]. injection Heq as <- _. apply free_symbol.
  - apply inv_record_set; [ | exact Ha ].
    intros base code [Heq | []]. injection Heq as <- _. apply free_symbol.
  - apply inv_block_deps; [ apply free_symbol | exact Ha ].
  - apply inv_add_facts; [ | exact Ha ]. unfold boxed_number_facts.
    destruct (boxed_number_kind contents); [ ok_list | ].
    destruct contents; ok_list.
  - apply inv_add_facts; [ ok_list | exact Ha ].
  - apply inv_add_facts; [ ok_list | exact Ha ].
  - apply inv_block_deps; [ apply free_symbol | exact Ha ].
Qed.

(* ======================================================================== *)
(* Code and calls                                                           *)
(* ======================================================================== *)

Lemma inv_create_known (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  (∀ x, x ∈ params -> free x) -> inv a ->
  inv (snd (create_known_arity_call_witness id params returns exn a)).
Proof.
  intros Hp Ha. unfold create_known_arity_call_witness, fresh. cbv zeta.
  cbn [fst snd]. apply inv_add_facts; [ ok_list | exact Ha ].
Qed.

Lemma inv_witness_deps (id : code_id) (returns : list node) (exn : node) :
  ∀ (l : list (list node * node)) (a : acc),
    (∀ first w x, (first, w) ∈ l -> x ∈ first -> free x) ->
    (∀ first w, (first, w) ∈ l -> free w) ->
    inv a -> inv (unknown_arity_witness_deps id returns exn l a).
Proof.
  induction l as [ | [first w] rest IH]; intros a Hp Hw Ha; [ exact Ha | ].
  cbn [unknown_arity_witness_deps]. cbv zeta.
  assert (Hf : ∀ x, x ∈ first -> free x)
    by (intros x Hx; exact (Hp first w x (or_introl eq_refl) Hx)).
  assert (Hw0 : free w) by exact (Hw first w (or_introl eq_refl)).
  assert (Hr : ∀ first' w' x, (first', w') ∈ rest -> x ∈ first' -> free x)
    by (intros f' w' x H Hx; exact (Hp f' w' x (or_intror H) Hx)).
  assert (Hrw : ∀ first' w', (first', w') ∈ rest -> free w')
    by (intros f' w' H; exact (Hw f' w' (or_intror H))).
  match goal with
  | |- context [add_facts ?L a] =>
    assert (Ha1 : inv (add_facts L a)) by (apply inv_add_facts;
                                            [ ok_list | exact Ha ])
  end.
  destruct rest as [ | [? next] rest'].
  - apply inv_add_facts; [ ok_list | exact Ha1 ].
  - unfold fresh. cbv beta iota zeta.
    apply IH; [ exact Hr | exact Hrw | ].
    apply inv_add_facts; [ ok_list | exact Ha1 ].
Qed.

Lemma inv_create_unknown (id : code_id) (is_tupled : bool) (ar : arity)
    (params returns : list node) (exn : node) (a : acc) :
  (∀ x, x ∈ params -> free x) -> inv a ->
  inv (snd (create_unknown_arity_call_witnesses id is_tupled ar params
              returns exn a)).
Proof.
  intros Hp Ha. unfold create_unknown_arity_call_witnesses.
  destruct is_tupled.
  - unfold create_unknown_arity_tupled_call_witnesses, fresh. cbv zeta.
    cbn [fst snd]. apply inv_add_facts; [ ok_list | exact Ha ].
  - unfold create_unknown_arity_non_tupled_call_witnesses. cbv zeta.
    match goal with
    | |- context [fresh_list ?n a] =>
      destruct (AW.fresh_list_spec n a) as (_ & _ & _ & _ & _ & Hm);
      pose proof (inv_fresh_list n a Ha) as Ha1;
      destruct (fresh_list n a) as [ws a1] eqn:E
    end.
    cbn [fst snd] in Hm, Ha1 |- *.
    apply inv_witness_deps; [ | | exact Ha1 ].
    + intros first w x H Hx. apply Hp. apply in_combine_l in H.
      exact (AW.group_in _ _ _ _ H Hx).
    + intros first w H. apply in_combine_r in H.
      destruct (Hm w H) as (m & -> & _). apply free_made.
Qed.

Lemma inv_prepare_code (id : code_id) (c : code) (a : acc) :
  inv a -> inv (prepare_code id c a).
Proof.
  intros Ha. unfold prepare_code.
  destruct (AW.fresh_list_spec (length (code_result_kinds c)) a)
    as (_ & _ & _ & _ & _ & Hr).
  pose proof (inv_fresh_list (length (code_result_kinds c)) a Ha) as Ha1.
  destruct (fresh_list _ a) as [returns a1] eqn:E1.
  cbn [fst snd] in Hr, Ha1.
  unfold fresh at 1. cbv beta iota.
  unfold fresh at 1. cbv beta iota.
  match goal with
  | |- context [fresh_list ?n ?b] =>
    destruct (AW.fresh_list_spec n b) as (_ & _ & _ & _ & _ & Hq);
    pose proof (inv_fresh_list n b Ha1) as Ha4;
    destruct (fresh_list n b) as [params a4] eqn:E4
  end.
  cbn [fst snd] in Hq, Ha4.
  assert (Hpv : ∀ x, x ∈ params -> free x).
  { intros x Hx. destruct (Hq x Hx) as (m & -> & _). apply free_made. }
  destruct (create_known_arity_call_witness id params returns _ a4)
    as [known a5] eqn:E5.
  assert (Ha5 : inv a5)
    by (change a5 with (snd (known, a5)); rewrite <- E5;
        apply inv_create_known; [ exact Hpv | exact Ha4 ]).
  match goal with
  | |- context [create_unknown_arity_call_witnesses ?i ?t ?ar params returns
                  ?x a5] =>
    pose proof (inv_create_unknown i t ar params returns x a5 Hpv Ha5)
      as Ha6;
    destruct (create_unknown_arity_call_witnesses i t ar params returns x a5)
      as [unknown a6] eqn:E6
  end.
  cbn [snd] in Ha6. cbv beta iota zeta.
  apply inv_add_code_dep.
  - cbn [AW.code_dep_nodes cd_my_closure cd_exn cd_return].
    intros x [<- | [<- | Hx]]; [ apply free_made | apply free_made | ].
    destruct (Hr x Hx) as (m & -> & _). apply free_made.
  - apply inv_add_fact; [ | exact Ha6 ].
    apply ok_free; [ intros x [<- | []]; apply free_code_id | exact I ].
Qed.

Lemma inv_prepare_binding (b : static_binding) (a : acc) :
  inv a -> inv (prepare_binding b a).
Proof. destruct b; [ apply inv_prepare_code | auto | auto ]. Qed.

Lemma inv_make_known (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  (∀ x, x ∈ exn :: returns -> free x) -> inv a ->
  inv (snd (make_known_arity_apply_widget env ap returns exn a))
  ∧ free (fst (make_known_arity_apply_widget env ap returns exn a)).
Proof.
  intros Hp Ha. unfold make_known_arity_apply_widget, fresh. cbv zeta.
  cbn [fst snd]. split; [ | apply free_made ].
  apply inv_cond_alias; [ apply free_made | ].
  apply inv_add_facts; [ ok_list | ].
  apply inv_note_simples. exact Ha.
Qed.

Lemma inv_apply_deps (returns : list node) (exn called : node) :
  ∀ (l : list (list simple * node)) (a : acc),
    (∀ x, x ∈ exn :: called :: returns -> free x) ->
    (∀ first w, (first, w) ∈ l -> free w) ->
    inv a -> inv (unknown_arity_apply_deps returns exn called l a).
Proof.
  intros l. induction l as [ | [first w] rest IH]; intros a Hp Hw Ha;
    [ exact Ha | ].
  cbn [unknown_arity_apply_deps]. cbv zeta.
  assert (Hrw : ∀ first' w', (first', w') ∈ rest -> free w')
    by (intros f' w' H; exact (Hw f' w' (or_intror H))).
  match goal with
  | |- context [add_facts ?L a] =>
    assert (Ha1 : inv (add_facts L a)) by (apply inv_add_facts;
                                            [ ok_list | exact Ha ])
  end.
  destruct rest as [ | [? next] rest'].
  - apply inv_add_facts; [ ok_list | exact Ha1 ].
  - unfold fresh. cbv beta iota zeta.
    apply IH; [ exact Hp | exact Hrw | ].
    apply inv_add_facts; [ ok_list | exact Ha1 ].
    exact (Hrw _ _ (or_introl eq_refl)).
Qed.

Lemma inv_make_unknown (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  (∀ x, x ∈ exn :: returns -> free x) -> inv a ->
  inv (snd (make_unknown_arity_apply_widget env ap returns exn a))
  ∧ free (fst (make_unknown_arity_apply_widget env ap returns exn a)).
Proof.
  intros Hp Ha. unfold make_unknown_arity_apply_widget, fresh.
  cbv beta iota zeta.
  match goal with
  | |- context [fresh_list ?n ?b] =>
    destruct (AW.fresh_list_spec n b) as (_ & _ & _ & _ & _ & Hq);
    assert (Hb : inv b) by (apply inv_add_facts; [ ok_list | exact Ha ]);
    pose proof (inv_fresh_list n b Hb) as Ha2;
    destruct (fresh_list n b) as [ws a2] eqn:E2
  end.
  cbn [fst snd] in Hq, Ha2. cbv beta iota zeta.
  match goal with
  | |- context [unknown_arity_apply_deps ?r ?x ?c ?l ?b] =>
    assert (H3 : inv (unknown_arity_apply_deps r x c l b))
  end.
  { apply inv_apply_deps; [ | | apply inv_note_simples; exact Ha2 ].
    - intros x [<- | [<- | Hx]]; [ apply Hp; left; reflexivity
                                 | apply free_made | ].
      apply Hp. right. exact Hx.
    - intros first w H. apply in_combine_r in H.
      destruct (Hq w H) as (m & -> & _). apply free_made. }
  destruct ws as [ | w ws']; cbn [fst snd]; split; try apply free_made.
  - exact H3.
  - apply inv_cond_alias; [ | exact H3 ].
    destruct (Hq w (or_introl eq_refl)) as (m & -> & _). apply free_made.
Qed.

Lemma inv_add_apply (env : traverse_env) (id : code_id)
    (callee widget : node) (only : bool) (a : acc) :
  free widget -> inv a -> inv (add_apply env id callee widget only a).
Proof.
  intros Hw Ha. unfold add_apply.
  destruct only; [ | apply inv_push_apply_dep; [ exact Hw | exact Ha ] ].
  unfold fresh. cbv beta iota zeta.
  apply inv_push_apply_dep; [ apply free_made | ].
  apply inv_add_facts; [ ok_list | exact Ha ].
Qed.

Lemma inv_call_kind (env : traverse_env) (ap : apply_expr) (exn_arg : node)
    (return_args : list node) (a : acc) :
  (∀ x, x ∈ exn_arg :: return_args -> free x) -> inv a ->
  inv (traverse_call_kind env ap exn_arg return_args a).
Proof.
  intros Hp Ha. unfold traverse_call_kind. cbv zeta.
  apply (inv_note_simple (ap_callee ap)) in Ha.
  destruct (ap_call_kind ap) as [id | | code_ids].
  - destruct (inv_make_known env ap return_args exn_arg
                (note_simple (ap_callee ap) a) Hp Ha) as [H4 Hcw].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg
                (note_simple (ap_callee ap) a)) as [cw a4] eqn:E4.
    cbn [fst snd] in *.
    assert (H5 : inv (add_fact (Fact_accessor cw
                                  (Field_call_witness
                                     Known_arity_code_pointer)
                                  (node_of_simple (ap_callee ap))) a4)).
    { apply inv_add_fact; [ | exact H4 ].
      apply ok_free; [ intros x [<- | []]; exact Hcw | exact I ]. }
    destruct (env_should_preserve_direct_calls env);
      try (apply inv_add_apply; [ exact Hcw | exact H5 ]).
    destruct (negb _); [ apply inv_add_apply; [ exact Hcw | exact H5 ]
                       | exact H5 ].
  - destruct (inv_make_unknown env ap return_args exn_arg
                (note_simple (ap_callee ap) a) Hp Ha) as [H4 Hcw].
    destruct (make_unknown_arity_apply_widget env ap return_args exn_arg
                (note_simple (ap_callee ap) a)) as [cw a4] eqn:E4.
    cbn [fst snd] in *.
    apply inv_add_fact; [ | exact H4 ].
    apply ok_free; [ intros x [<- | []]; exact Hcw | exact I ].
  - destruct (inv_make_known env ap return_args exn_arg
                (note_simple (ap_callee ap) a) Hp Ha) as [H4 Hcw].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg
                (note_simple (ap_callee ap) a)) as [cw a4] eqn:E4.
    cbn [fst snd] in *.
    apply inv_add_fact; [ | exact H4 ].
    apply ok_free; [ intros x [<- | []]; exact Hcw | exact I ].
Qed.

Lemma inv_traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  penv env -> inv a -> inv (traverse_apply env ap a).
Proof.
  intros Henv Ha. unfold traverse_apply. cbv zeta.
  destruct (inv_find_cont env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a Henv Ha) as [H1 Hr].
  destruct (find_cont env (ap_continuation ap) _ a) as [ra a1] eqn:E1.
  cbn [fst snd] in *.
  destruct (inv_find_cont env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) a1 Henv H1) as [H2 He].
  destruct (find_cont env (ap_exn_continuation ap) _ a1) as [ep a2] eqn:E2.
  cbn [fst snd] in *.
  destruct ep as [ | p ps].
  - unfold fresh. cbv beta iota.
    apply inv_call_kind; [ | exact H2 ].
    intros y [<- | Hy]; [ apply free_made | exact (Hr y Hy) ].
  - cbv beta iota zeta.
    apply inv_call_kind.
    + intros x [<- | Hx]; [ apply He; left; reflexivity | exact (Hr x Hx) ].
    + apply inv_add_facts; [ | apply inv_note_simples; exact H2 ].
      ok_list. apply He. right. eapply in_combine_l. eassumption.
Qed.

Lemma inv_apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc) :
  penv env -> inv a -> inv (apply_cont_deps env ac a).
Proof.
  intros Henv Ha. unfold apply_cont_deps. cbv zeta.
  destruct (inv_find_cont env (ac_continuation ac) (length (ac_args ac)) a
              Henv Ha) as [H1 Hp].
  destruct (find_cont env (ac_continuation ac) _ a) as [ps a1] eqn:E1.
  cbn [fst snd] in *.
  apply inv_add_facts; [ | apply inv_note_simples; exact H1 ].
  ok_list.
Qed.

Lemma inv_traverse_switch (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  penv env -> inv a -> inv (traverse_switch env scrutinee arms a).
Proof.
  intros Henv Ha. unfold traverse_switch. cbv zeta.
  apply inv_fold; [ | apply inv_cond_any_usage; exact Ha ].
  intros a' [i ac] _ Ha'. apply inv_apply_cont_deps; assumption.
Qed.

Section Params.

Hypothesis Hnl : NoDup (map fst (let_bindings e0)).
Hypothesis Hlp :
  ∀ v, v ∈ map fst (let_bindings e0) -> v ∈ param_variables e0 -> False.

Lemma inv_exn_handler_deps (env : traverse_env) (params : list variable)
    (md : handler_metadata) (body : expr) (a : acc) :
  (∀ p, p ∈ params -> p ∈ param_variables e0) -> inv a ->
  inv (exn_handler_deps env (Handler params md body) a).
Proof.
  intros Hp Ha. unfold exn_handler_deps.
  destruct params as [ | p ps]; [ exact Ha | ].
  destruct (hm_is_exn_handler md); [ | exact Ha ].
  apply inv_cond_any_usage. apply inv_add_fact; [ | exact Ha ].
  apply ok_free; [ | exact I ]. intros x [<- | []].
  apply (free_param Hlp). apply Hp. left. reflexivity.
Qed.

Lemma inv_code_aliases (my_closure : variable) (params : list variable)
    (cd : code_dep) (a : acc) :
  my_closure ∈ param_variables e0 ->
  (∀ p, p ∈ params -> p ∈ param_variables e0) -> inv a ->
  inv (add_facts
         (Fact_alias (Node_variable my_closure) (cd_my_closure cd)
          :: map (fun '(p, arg) => Fact_alias (Node_variable p) arg)
               (combine params (cd_params cd)))
         a).
Proof.
  intros Hm Hp Ha. apply inv_add_facts; [ | exact Ha ]. ok_list.
  - exact (free_param Hlp _ Hm).
  - apply (free_param Hlp). apply Hp. eapply in_combine_l. eassumption.
Qed.

(* ======================================================================== *)
(* The traversal                                                            *)
(* ======================================================================== *)

Definition sub (e : expr) : Prop :=
  (∀ p, p ∈ let_bindings e -> p ∈ let_bindings e0)
  ∧ (∀ v, v ∈ param_variables e -> v ∈ param_variables e0).

Definition P_t (e : expr) : Prop :=
  ∀ (env : traverse_env) (a : acc),
    sub e -> penv env -> env_set_closures env = set_closures e0 -> inv a ->
    inv (traverse env e a).

Lemma penv_add_cont (env : traverse_env) (k : continuation)
    (params : list variable) :
  (∀ p, p ∈ params -> p ∈ param_variables e0) -> penv env ->
  penv (add_cont env k (map Node_variable params)).
Proof.
  intros Hp Henv k' ps x [Heq | Hin] Hx.
  - injection Heq as <- <-. apply in_map_iff in Hx as (p & <- & Hp').
    exact (free_param Hlp p (Hp p Hp')).
  - exact (Henv _ _ _ Hin Hx).
Qed.

Lemma add_rec_conts_sc (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) :
  env_set_closures (add_rec_conts env handlers) = env_set_closures env.
Proof.
  unfold add_rec_conts. induction handlers as [ | [k h] hs IH];
    [ reflexivity | exact IH ].
Qed.

Lemma penv_add_rec_conts (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) :
  (∀ k params md body, (k, Handler params md body) ∈ handlers ->
     ∀ p, p ∈ params -> p ∈ param_variables e0) ->
  penv env -> penv (add_rec_conts env handlers).
Proof.
  intros Hp Henv. unfold add_rec_conts.
  induction handlers as [ | [k [params md body]] hs IH]; cbn [fold_right];
    [ exact Henv | ].
  apply penv_add_cont.
  - exact (Hp k params md body (or_introl eq_refl)).
  - apply IH. intros k' ps md' b' H. exact (Hp k' ps md' b' (or_intror H)).
Qed.

Lemma inv_traverse :
  (∀ e : expr, P_t e)
  ∧ ∀ h : continuation_handler,
      match h with Handler _ _ handler => P_t handler end.
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind; unfold P_t.
  - intros x k d body IH env a [HL HP] Henv Hsc Ha.
    cbn [let_bindings param_variables] in HL, HP. cbn [traverse].
    apply IH; [ | exact Henv | exact Hsc | ].
    + split; [ intros p Hp; apply HL; right; exact Hp | exact HP ].
    + apply (inv_traverse_named Hnl); [ exact Hsc | | exact Ha ].
      apply HL. left. reflexivity.
  - intros bindings body Hcodes IH env a [HL HP] Henv Hsc Ha.
    cbn [let_bindings param_variables] in HL, HP. cbn [traverse].
    apply IH; [ | exact Henv | exact Hsc | ].
    { split; intros; [ apply HL | apply HP ]; apply in_or_app; right;
        assumption. }
    apply inv_fold.
    { intros a' bd Hbd Ha'. destruct bd as [id c | id | s const];
        cbn [traverse_binding]; [ | exact Ha' | exact Ha' ].
      pose proof (Hcodes id c Hbd) as Hc.
      destruct c as [params mc ret exn md cbody]. cbn [code_body] in Hc.
      cbn [traverse_code].
      destruct (lookup id (acc_code_deps a')) as [cd | ] eqn:Ecd;
        [ | exact Ha' ].
      assert (Hpar : ∀ v, v ∈ code_param_variables
                                (Code params mc ret exn md cbody) ->
                       v ∈ param_variables e0).
      { intros v Hv. apply HP. apply in_or_app. left. apply in_flat_map.
        exists (Bind_code id (Code params mc ret exn md cbody)).
        split; [ exact Hbd | exact Hv ]. }
      cbn [code_param_variables] in Hpar.
      cbv zeta. apply Hc; [ | | exact Hsc | ].
      - split.
        + intros p Hp. apply HL. apply in_or_app. left. apply in_flat_map.
          exists (Bind_code id (Code params mc ret exn md cbody)).
          split; [ exact Hbd | exact Hp ].
        + intros v Hv. apply Hpar. right. apply in_or_app. right. exact Hv.
      - intros k' ps x Hk Hx.
        cbn [code_env FlambdaReaperTraversal.env_conts] in Hk.
        pose proof (proj1 (proj2 Ha') id cd (lookup_in _ _ _ Ecd)) as Hn.
        destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
          apply Hn; cbn [AW.code_dep_nodes].
        + right. left. destruct Hx as [<- | []]. reflexivity.
        + right. right. exact Hx.
      - apply inv_code_aliases; [ apply Hpar; left; reflexivity | | exact Ha' ].
        intros p Hp. apply Hpar. right. apply in_or_app. left. exact Hp. }
    apply inv_fold.
    { intros a' bd _ Ha'. apply inv_static_binding. exact Ha'. }
    apply inv_fold; [ | exact Ha ].
    intros a' bd _ Ha'. apply inv_prepare_binding. exact Ha'.
  - intros k h body IHh IH env a [HL HP] Henv Hsc Ha.
    destruct h as [params md handler].
    cbn [let_bindings param_variables handler_let_bindings
         handler_param_variables] in HL, HP.
    cbn [traverse traverse_handler handler_params]. cbv zeta.
    apply IH; [ | | exact Hsc | ].
    + split; intros; [ apply HL | apply HP ]; apply in_or_app; right;
        assumption.
    + apply penv_add_cont; [ | exact Henv ].
      intros p Hp. apply HP. apply in_or_app. left. apply in_or_app.
      left. exact Hp.
    + apply inv_exn_handler_deps.
      { intros p Hp. apply HP. apply in_or_app. left. apply in_or_app.
        left. exact Hp. }
      apply IHh; [ | exact Henv | exact Hsc | exact Ha ].
      split; intros; [ apply HL | apply HP ]; apply in_or_app; left;
        [ assumption | apply in_or_app; right; assumption ].
  - intros handlers body IHh IH env a [HL HP] Henv Hsc Ha.
    cbn [let_bindings param_variables] in HL, HP.
    cbn [traverse]. cbv zeta.
    assert (Hsc' : env_set_closures (add_rec_conts env handlers)
                   = set_closures e0)
      by (rewrite add_rec_conts_sc; exact Hsc).
    assert (Hpar : ∀ k params md body, (k, Handler params md body) ∈ handlers
                     -> ∀ p, p ∈ params ++ param_variables body ->
                          p ∈ param_variables e0).
    { intros k params md b Hk p Hp. apply HP. apply in_or_app. left.
      apply in_flat_map. exists (k, Handler params md b).
      split; [ exact Hk | exact Hp ]. }
    assert (Henv' : penv (add_rec_conts env handlers)).
    { apply penv_add_rec_conts; [ | exact Henv ].
      intros k params md b Hk p Hp. apply (Hpar k params md b Hk).
      apply in_or_app. left. exact Hp. }
    apply IH; [ | exact Henv' | exact Hsc' | ].
    + split; intros; [ apply HL | apply HP ]; apply in_or_app; right;
        assumption.
    + apply inv_fold; [ | exact Ha ].
      intros a' [k h] Hkh Ha'. pose proof (IHh k h Hkh) as Hh.
      destruct h as [params md handler]. cbn [traverse_handler].
      apply Hh; [ | exact Henv' | exact Hsc' | exact Ha' ].
      split.
      * intros p Hp. apply HL. apply in_or_app. left. apply in_flat_map.
        exists (k, Handler params md handler). split; [ exact Hkh | exact Hp ].
      * intros v Hv. apply (Hpar k params md handler Hkh).
        apply in_or_app. right. exact Hv.
  - intros ap env a _ Henv _ Ha. apply inv_traverse_apply; assumption.
  - intros ac env a _ Henv _ Ha. apply inv_apply_cont_deps; assumption.
  - intros scrutinee arms env a _ Henv _ Ha.
    apply inv_traverse_switch; assumption.
  - intros env a _ _ _ Ha. exact Ha.
  - intros params md handler IH. exact IH.
Qed.

End Params.

(* ======================================================================== *)
(* The delayed deps                                                         *)
(* ======================================================================== *)

Lemma ok_alias_for_caller (caller : option code_id) (from to_ : node) :
  free to_ -> fact_ok (alias_for_caller caller from to_).
Proof.
  intros H. destruct caller;
    (apply ok_free; [ intros x [<- | []]; exact H | exact I ]).
Qed.

Lemma inv_add_closure_dep (code_deps : finite_map code_dep)
    (d : closure_dep) (a : acc) :
  (∀ id cd, (id, cd) ∈ code_deps -> ∀ x, x ∈ AW.code_dep_nodes cd -> free x)
  -> free (cld_let_bound_name_of_the_closure d) -> inv a ->
  inv (add_closure_dep code_deps d a).
Proof.
  intros Hcd Hd Ha. unfold add_closure_dep.
  destruct (lookup (cld_closure_code_id d) code_deps) as [cd | ] eqn:E.
  - assert (Hm : free (cd_my_closure cd))
      by exact (Hcd _ cd (lookup_in _ _ _ E) _ (or_introl eq_refl)).
    apply inv_add_facts; [ | exact Ha ].
    destruct (cd_unknown_arity_call_witnesses cd); ok_list.
  - unfold fresh. cbv beta iota zeta.
    apply inv_add_facts; [ ok_list | exact Ha ].
Qed.

Lemma inv_resolve (unit_symbols : list symbol) (a : acc) :
  inv a -> inv (resolve_delayed_deps unit_symbols a).
Proof.
  intros Ha. pose proof Ha as (H1 & H2 & H3 & H4).
  unfold resolve_delayed_deps. cbv zeta.
  match goal with
  | |- context [add_facts (flat_map ?g (acc_apply_deps a)) a] =>
    assert (Hb : inv (add_facts (flat_map g (acc_apply_deps a)) a));
    [ | set (b := add_facts (flat_map g (acc_apply_deps a)) a) in * ]
  end.
  { apply inv_add_facts; [ | exact Ha ].
    intros fct Hin. apply in_flat_map in Hin as (d & Hd & Hin).
    pose proof (H3 d Hd) as Hw.
    unfold apply_dep_facts in Hin. cbv zeta in Hin.
    destruct (lookup (ad_apply_code_id d) (acc_code_deps a))
      as [cd | ] eqn:E.
    - assert (Hm : free (cd_my_closure cd))
        by exact (H2 _ cd (lookup_in _ _ _ E) _ (or_introl eq_refl)).
      destruct Hin as [<- | [<- | []]]; apply ok_alias_for_caller;
        assumption.
    - destruct (ad_function_containing_apply_expr d);
        cbv beta iota in Hin; revert fct Hin; ok_list. }
  apply inv_add_facts; [ ok_list | ].
  apply inv_fold; [ | exact Hb ].
  intros b' d Hd Hb'. apply inv_add_closure_dep; [ exact H2 | | exact Hb' ].
  exact (H4 d Hd).
Qed.

End Slots.

(* ======================================================================== *)
(* The unit                                                                 *)
(* ======================================================================== *)

Lemma unit_inv (flag : reaper_preserve_direct_calls) (u : unit_interface)
    (e : expr) :
  variables_bound_once e = true -> inv e (traverse_unit_acc flag u e).
Proof.
  intros Honce.
  pose proof Honce as Ho. unfold variables_bound_once in Ho.
  apply distinct_NoDup in Ho.
  assert (Hnl : NoDup (map fst (let_bindings e)))
    by exact (NoDup_app_remove_r _ _ Ho).
  assert (Hlp : ∀ v, v ∈ map fst (let_bindings e) -> v ∈ param_variables e ->
                False)
    by (intros v H1 H2;
        exact (FlambdaReaperLeastProofs.BoundOnce.nodup_app_disjoint _ _ v Ho
                 H1 H2)).
  unfold traverse_unit_acc. cbv zeta.
  set (a1 := add_fact (Fact_any_source Node_all_constants) empty_acc).
  assert (H1 : inv e a1).
  { apply inv_add_fact.
    - apply ok_free; [ intros x [<- | []]; apply free_consts | exact I ].
    - split; [ intros ? [] | split; [ intros ? ? [] | ] ].
      split; intros ? []. }
  cbn [fresh]. cbv beta iota.
  match goal with
  | |- inv _ (resolve_delayed_deps _ (add_fact _ (traverse ?env e ?a))) =>
    assert (Henv : penv e env); [ | assert (Ha : inv e a) ]
  end.
  { intros k ps x Hk Hx. cbn in Hk.
    destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
      destruct Hx as [<- | []]; apply free_made. }
  { apply inv_add_fact; [ | exact H1 ].
    apply ok_free; [ intros x [] | exact I ]. }
  pose proof (proj1 (inv_traverse e Hnl Hlp) e _ _
                (conj (fun p H => H) (fun p H => H)) Henv eq_refl Ha) as H2.
  apply inv_resolve. apply inv_add_fact; [ | exact H2 ].
  apply ok_free; [ intros x [] | exact I ].
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem slot_read_only : slot_read_only_statement.
Proof.
  intros flag u e v d w c Honce Hd Hs.
  destruct (unit_inv flag u e Honce) as (Hok & _).
  rewrite traverse_unit_acc_facts.
  apply CH.only_read_into_of.
  intros fct Hin Hv. exact (proj1 (Hok fct Hin) v Hv d w c Hd Hs).
Qed.

Theorem slot_reads_projected : slot_reads_projected_statement.
Proof.
  intros flag u e to_ w b Honce Hin.
  destruct (unit_inv flag u e Honce) as (Hok & _).
  rewrite traverse_unit_acc_facts in Hin.
  cbn [facts_of_list f_accessor] in Hin.
  apply in_flat_map in Hin as (fct & Hf & Hin).
  destruct fct; cbn in Hin; try contradiction.
  destruct Hin as [Heq | []]. injection Heq as -> -> ->.
  exact (proj2 (Hok _ Hf)).
Qed.
