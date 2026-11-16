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

(* Proofs of the statements of FlambdaSharedReadsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Arith Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaBindersSpec
  FlambdaNominalSpec FlambdaNominalProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaStaticsInScopeSpec FlambdaUnusedStaticsSpec
  FlambdaHeapSpec FlambdaClosuresProofs FlambdaContextWhereSpec
  FlambdaContextWhereProofs FlambdaImportsSpec FlambdaObserversSpec
  FlambdaScopedProofs FlambdaEquivarianceTactic FlambdaSharedReadsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Places and classes                                                       *)
(* ======================================================================== *)

Lemma same_place_eq (p p' : static_place) : same_place p p' = true -> p = p'.
Proof.
  destruct p as [slot | field], p' as [slot' | field']; simpl;
    try discriminate.
  - intros H. apply Pos.eqb_eq in H. subst. reflexivity.
  - intros H. apply Z.eqb_eq in H. subst. reflexivity.
Qed.

Lemma same_place_refl (p : static_place) : same_place p p = true.
Proof.
  destruct p as [slot | field]; simpl;
    [ apply Pos.eqb_refl | apply Z.eqb_refl ].
Qed.

Lemma same_class_eq (c c' : place_class) : same_class c c' = true -> c = c'.
Proof.
  destruct c as [first x], c' as [first' x']. unfold same_class. simpl.
  intros H. apply andb_true_iff in H as [Hfirst Hx].
  apply Pos.eqb_eq in Hfirst. apply Pos.eqb_eq in Hx. subst. reflexivity.
Qed.

Lemma same_class_refl (c : place_class) : same_class c c = true.
Proof. unfold same_class. rewrite !Pos.eqb_refl. reflexivity. Qed.

(* The class of a place is that of an entry for it. *)
Lemma class_of_in (held : held_places) (s : symbol) (p : static_place)
    (c : place_class) :
  class_of held s p = Some c -> (s, p, c) ∈ held.
Proof.
  induction held as [ | [[s' p'] c'] held IH]; simpl; [ discriminate | ].
  destruct (Pos.eqb s s' && same_place p p') eqn:Hsame.
  - intros [= <-]. apply andb_true_iff in Hsame as [Hs Hp].
    apply Pos.eqb_eq in Hs. apply same_place_eq in Hp. subst. left.
    reflexivity.
  - intros H. right. exact (IH H).
Qed.

(* The first read made of a class is a read made. *)
Lemma read_before_in (made : reads_made) (c : place_class) (v : variable) :
  read_before made c = Some v -> (c, v) ∈ made.
Proof.
  induction made as [ | [c' v'] made IH]; simpl; [ discriminate | ].
  destruct (same_class c c') eqn:Hsame.
  - intros [= <-]. apply same_class_eq in Hsame. subst. left. reflexivity.
  - intros H. right. exact (IH H).
Qed.

(* ======================================================================== *)
(* What a constant holds, and what it makes                                 *)
(* ======================================================================== *)

Lemma eval_simples_nth (vars : env) :
  ∀ (args : list simple) (vs : list value) (n : nat) (a : simple),
    eval_simples vars args = Some vs ->
    nth_error args n = Some a ->
    nth_error vs n = eval_simple vars a.
Proof.
  induction args as [ | a' args IH]; intros vs n a; simpl.
  - intros _ H. destruct n; discriminate H.
  - destruct (eval_simple vars a') as [v | ] eqn:Ha'; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. destruct n as [ | n]; simpl.
    + intros [= <-]. symmetry. exact Ha'.
    + apply IH. reflexivity.
Qed.

Theorem place_held_value : place_held_value_statement.
Proof.
  intros vars const allocated p a Hconst Hheld.
  destruct p as [slot | field];
    destruct const as [code slots | fslot syms decls slots | tag fields
      | contents | bytes | elements];
    simpl in Hheld; try discriminate; simpl in Hconst.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl in Hconst. injection Hconst as <-. simpl.
    exact (eval_value_slots_lookup vars slot a slots vs Hslots Hheld).
  - destruct (lookup fslot decls); [ | discriminate ]. simpl in Hconst.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl in Hconst. injection Hconst as <-. simpl.
    exact (eval_value_slots_lookup vars slot a slots vs Hslots Hheld).
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    simpl in Hconst. injection Hconst as <-. simpl.
    destruct (0 <=? field)%Z; [ | discriminate ].
    exact (eval_simples_nth vars fields vs _ a Hfields Hheld).
Qed.

(* ======================================================================== *)
(* A read                                                                   *)
(* ======================================================================== *)

(* The forms of a read. *)
Lemma read_of_cases (d : named) (s : symbol) (p : static_place) :
  read_of d = Some (s, p) ->
  (∃ slot : value_slot,
    d = Project_value_slot slot (Symbol s) ∧ p = Place_slot slot)
  ∨ (∃ (from : function_slot) (slot : value_slot),
      d = Project_value_slot_from from slot (Symbol s) ∧ p = Place_slot slot)
  ∨ (∃ (kind : block_access_kind) (mut : mutability) (field : Z),
      d = Prim (Unary (Block_load kind mut field)) [Symbol s]
      ∧ p = Place_field field).
Proof.
  destruct d as
    [a | q args | code slots | fslot decls slots | slot closure
    | from slot closure | move_from move_to closure];
    simpl; try discriminate.
  - destruct q as [u | b | va | h | rp]; try discriminate.
    destruct u; try discriminate.
    destruct mut, args as [ | [x | sym | c] [ | arg args]]; simpl;
      try discriminate;
      intros [= <- <-]; right; right; eexists _, _, _; split; reflexivity.
  - destruct closure as [x | sym | c]; try discriminate.
    intros [= <- <-]. left. exists slot. split; reflexivity.
  - destruct closure as [x | sym | c]; try discriminate.
    intros [= <- <-]. right. left. exists from, slot. split; reflexivity.
Qed.

Theorem read_of_place : read_of_place_statement.
Proof.
  intros statics vars d s p v Hread Hd.
  destruct (read_of_cases d s p Hread)
    as [(slot & -> & ->)
       | [(from & slot & -> & ->) | (kind & mut & field & -> & ->)]];
    simpl in Hd.
  - unfold closure_of in Hd.
    destruct (lookup s (static_values statics)) as [allocated | ];
      [ | discriminate ].
    exists allocated. split; [ reflexivity | ].
    destruct allocated as [code vs | fslot syms decls vs | | | | | ];
      try discriminate; simpl.
    + exact Hd.
    + destruct (lookup fslot decls); [ exact Hd | discriminate ].
  - unfold project_value_slot, closure_in_set in Hd.
    destruct (lookup s (static_values statics)) as [allocated | ];
      [ | discriminate ].
    exists allocated. split; [ reflexivity | ].
    destruct allocated as [code vs | fslot syms decls vs | | | | | ];
      try discriminate; simpl; simpl in Hd.
    + destruct (Pos.eqb unnamed_function_slot from);
        [ exact Hd | discriminate ].
    + destruct (lookup fslot decls); [ | discriminate ]. simpl in Hd.
      destruct (Pos.eqb fslot from); [ exact Hd | discriminate ].
  - rewrite eval_prim_unary in Hd. cbn [eval_unary_primitive] in Hd.
    destruct kind as [tag size field_kind]. cbn [eval_block_load] in Hd.
    unfold block_of in Hd.
    destruct (lookup s (static_values statics)) as [allocated | ];
      cbv beta iota in Hd; [ | discriminate Hd ].
    exists allocated. split; [ reflexivity | ].
    destruct allocated as [ | | block_tag fields | | | | ];
      cbv beta iota in Hd; try discriminate Hd.
    cbn [place_value].
    destruct (0 <=? field)%Z;
      [ | rewrite andb_false_r in Hd; discriminate Hd ].
    destruct (known_is tag block_tag && known_is size _ && true);
      [ | discriminate Hd ].
    destruct (nth_error fields (Z.to_nat field)) as [loaded | ];
      cbv beta iota in Hd; [ | discriminate Hd ].
    destruct
      (match field_kind with
       | Any_value => kind_eqb (kind_of_value loaded) Value
       | Immediate => is_tagged_immediate loaded
       end);
      [ | discriminate Hd ].
    injection Hd as <-. reflexivity.
Qed.

Theorem read_of_exec : read_of_exec_statement.
Proof.
  intros statics hp hp' rd rd' vars d s p v Hread.
  destruct (read_of_cases d s p Hread)
    as [(slot & -> & ->)
       | [(from & slot & -> & ->) | (kind & mut & field & -> & ->)]].
  - rewrite (exec_named_not_prim statics hp rd vars _)
      by (intros q args; discriminate).
    destruct (eval_named statics vars _) as [w | ]; [ | discriminate ].
    simpl. intros [= <- <-]. split; reflexivity.
  - rewrite (exec_named_not_prim statics hp rd vars _)
      by (intros q args; discriminate).
    destruct (eval_named statics vars _) as [w | ]; [ | discriminate ].
    simpl. intros [= <- <-]. split; reflexivity.
  - rewrite
      (exec_named_no_location statics hp rd vars _ [Symbol s] [Value_symbol s])
      by first
           [ reflexivity
           | intros q; discriminate
           | intros l [Hl | []]; discriminate Hl ].
    destruct (eval_named statics vars _) as [w | ]; [ | discriminate ].
    simpl. intros [= <- <-]. split; reflexivity.
Qed.

(* ======================================================================== *)
(* A group                                                                  *)
(* ======================================================================== *)

(* A symbol that the group does not bind is bound after it as before
   it. *)
Lemma eval_static_bindings_other (vars : env) (s : symbol) :
  ∀ (bindings : list static_binding) (statics statics' : static_env),
    group_const s bindings = None ->
    eval_static_bindings vars bindings statics = Some statics' ->
    lookup s (static_values statics') = lookup s (static_values statics).
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics'; simpl.
  - intros _ [= <-]. reflexivity.
  - intros Hnone.
    destruct (lookup id (static_code statics)); [ discriminate | ].
    intros Heval. rewrite (IH _ _ Hnone Heval). reflexivity.
  - intros Hnone Heval. exact (IH _ _ Hnone Heval).
  - destruct (Pos.eqb s sym) eqn:Hs; [ discriminate | ]. intros Hnone.
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ]; simpl;
      [ | discriminate ].
    intros Heval. rewrite (IH _ _ Hnone Heval). simpl. rewrite Hs.
    reflexivity.
Qed.

(* A symbol that the group binds is bound after it to what its
   constant makes. *)
Lemma eval_static_bindings_const (vars : env) (s : symbol)
    (const : static_const) :
  ∀ (bindings : list static_binding) (statics statics' : static_env),
    group_const s bindings = Some const ->
    eval_static_bindings vars bindings statics = Some statics' ->
    ∃ allocated : statically_allocated_value,
      eval_static_const vars const = Some allocated
      ∧ lookup s (static_values statics') = Some allocated.
Proof.
  induction bindings as [ | [id c | id | sym const'] bindings IH];
    intros statics statics'; simpl.
  - discriminate.
  - intros Hconst.
    destruct (lookup id (static_code statics)); [ discriminate | ].
    intros Heval. exact (IH _ _ Hconst Heval).
  - intros Hconst Heval. exact (IH _ _ Hconst Heval).
  - destruct (Pos.eqb s sym) eqn:Hs.
    + intros [= ->]. apply Pos.eqb_eq in Hs. subst sym.
      destruct (lookup s (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ]; simpl;
        [ | discriminate ].
      intros Heval. exists allocated. split; [ reflexivity | ].
      destruct (eval_static_bindings_extend vars bindings _ _ Heval)
        as [_ Hvalues].
      apply Hvalues. simpl. rewrite Pos.eqb_refl. reflexivity.
    + intros Hconst.
      destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const') as [allocated | ]; simpl;
        [ | discriminate ].
      intros Heval. exact (IH _ _ Hconst Heval).
Qed.

(* After a group, the place of an entry of the group holds the value
   of the variable of its class. *)
Lemma entry_of_group_value (vars : env) (bindings : list static_binding)
    (statics statics' : static_env) (s : symbol) (p : static_place)
    (first : symbol) (x : variable) :
  entry_of_group bindings (s, p, (first, x)) = true ->
  eval_static_bindings vars bindings statics = Some statics' ->
  first_symbol bindings = Some first
  ∧ ∃ allocated : statically_allocated_value,
      lookup s (static_values statics') = Some allocated
      ∧ place_value allocated p = lookup x vars.
Proof.
  unfold entry_of_group. intros Hentry Heval.
  destruct (first_symbol bindings) as [first' | ]; [ | discriminate ].
  destruct (group_const s bindings) as [const | ] eqn:Hconst;
    [ | discriminate ].
  apply andb_true_iff in Hentry as [Hfirst Hheld].
  apply Pos.eqb_eq in Hfirst. subst first'. split; [ reflexivity | ].
  destruct (place_held const p) as [[x' | sym | c] | ] eqn:Hplace;
    try discriminate.
  apply Pos.eqb_eq in Hheld. subst x'.
  destruct (eval_static_bindings_const vars s const bindings statics statics'
              Hconst Heval)
    as (allocated & Hallocated & Hlookup).
  exists allocated. split; [ exact Hlookup | ].
  exact (place_held_value vars const allocated p (Var x) Hallocated Hplace).
Qed.

(* An entry that has to do with the group, where the group passes the
   check, is of the group. *)
Lemma group_holds_entry (held : held_places) (bindings : list static_binding)
    (entry : symbol * static_place * place_class) :
  group_holds held bindings = true ->
  entry ∈ held ->
  entry_meets_group bindings entry = true ->
  entry_of_group bindings entry = true.
Proof.
  unfold group_holds. intros Hgroup Hin Hmeets.
  pose proof (proj1 (forallb_forall _ _) Hgroup entry Hin) as Hentry.
  cbv beta in Hentry. rewrite Hmeets in Hentry. exact Hentry.
Qed.

Theorem group_keeps_classes : group_keeps_classes_statement.
Proof.
  intros held vars bindings statics statics' Hgroup Hagree Heval s s' p p'
    [first x] allocated allocated' v v' Hin Hin' Hs Hs' Hv Hv'.
  destruct (entry_meets_group bindings (s, p, (first, x))) eqn:Hmeets.
  - (* The group binds all the symbols of the class. *)
    destruct (entry_of_group_value vars bindings statics statics' s p first x
                (group_holds_entry held bindings _ Hgroup Hin Hmeets) Heval)
      as (Hfirst & a & Ha & Hplace).
    assert (Hmeets' :
      entry_meets_group bindings (s', p', (first, x)) = true).
    { unfold entry_meets_group. rewrite Hfirst, Pos.eqb_refl.
      apply orb_true_r. }
    destruct (entry_of_group_value vars bindings statics statics' s' p' first
                x (group_holds_entry held bindings _ Hgroup Hin' Hmeets')
                Heval)
      as (_ & a' & Ha' & Hplace').
    rewrite Hs in Ha. injection Ha as <-.
    rewrite Hs' in Ha'. injection Ha' as <-.
    rewrite Hv in Hplace. rewrite Hv' in Hplace'.
    rewrite <- Hplace in Hplace'. injection Hplace' as ->. reflexivity.
  - destruct (entry_meets_group bindings (s', p', (first, x))) eqn:Hmeets'.
    + (* It cannot bind one and not the other. *)
      destruct (entry_of_group_value vars bindings statics statics' s' p'
                  first x
                  (group_holds_entry held bindings _ Hgroup Hin' Hmeets')
                  Heval)
        as (Hfirst & _).
      unfold entry_meets_group in Hmeets.
      rewrite Hfirst, Pos.eqb_refl, orb_true_r in Hmeets. discriminate.
    + (* It binds neither. *)
      unfold entry_meets_group in Hmeets, Hmeets'.
      apply orb_false_iff in Hmeets as [Hnone _].
      apply orb_false_iff in Hmeets' as [Hnone' _].
      destruct (group_const s bindings) eqn:Hconst; [ discriminate | ].
      destruct (group_const s' bindings) eqn:Hconst'; [ discriminate | ].
      rewrite (eval_static_bindings_other vars s bindings statics statics'
                 Hconst Heval)
        in Hs.
      rewrite (eval_static_bindings_other vars s' bindings statics statics'
                 Hconst' Heval)
        in Hs'.
      exact (Hagree s s' p p' (first, x) allocated allocated' v v' Hin Hin'
               Hs Hs' Hv Hv').
Qed.

(* ======================================================================== *)
(* What a run keeps                                                         *)
(* ======================================================================== *)

(* A step that changes the static environment is that of a group. *)
Lemma step_statics (s s' : state) :
  step s = Step s' ->
  state_statics s' = state_statics s
  ∨ ∃ (bindings : list static_binding) (body : expr),
      state_expr s = Let_static bindings body
      ∧ eval_static_bindings (state_vars s) bindings (state_statics s)
        = Some (state_statics s').
Proof.
  intros Hstep. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ] eqn:Hexpr.
  - rewrite step_let_exec in Hstep.
    destruct (exec_named statics (state_heap s) (state_reads s) vars d) as
      [[[v hp'] rd'] | ];
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. left. reflexivity.
  - rewrite step_let_static in Hstep.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-. right. exists bindings, body.
    split; [ reflexivity | exact Heval ].
  - rewrite step_let_cont_non_recursive in Hstep. injection Hstep as <-.
    left. reflexivity.
  - rewrite step_let_cont_recursive in Hstep.
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. left. reflexivity.
  - rewrite step_apply in Hstep.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    left. exact (call_function_statics _ _ _ _ _ _ _ _ _ rd_ _ Hstep).
  - rewrite step_apply_cont in Hstep.
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep.
    left. exact (apply_cont_value_statics _ _ _ _ rd_ _ Hstep).
  - rewrite step_switch in Hstep.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms); [ | discriminate ].
    injection Hstep as <-. left. reflexivity.
  - rewrite step_invalid in Hstep. discriminate.
Qed.

Section Held.

Variable held : held_places.

(* The terms of a state pass the check. *)
Definition groups_inv : state -> Prop :=
  terms_inv (groups_hold held) (handler_groups_hold held)
    (code_groups_hold held).

Lemma groups_inv_step (s s' : state) :
  groups_inv s -> step s = Step s' -> groups_inv s'.
Proof.
  apply terms_inv_step.
  - intros x xkind d body H. exact H.
  - intros bindings body H. simpl in H.
    apply andb_true_iff in H as [H Hbody].
    apply andb_true_iff in H as [_ Hcode].
    split; [ exact Hbody | ]. intros id c Hin.
    exact (proj1 (forallb_forall _ _) Hcode (Bind_code id c) Hin).
  - intros k h body H. simpl in H. apply andb_true_iff in H. tauto.
  - intros handlers body H. simpl in H. apply andb_true_iff in H. tauto.
  - intros ac. reflexivity.
  - intros params metadata handler H. exact H.
  - intros [params my_closure ret exn metadata body] H. exact H.
Qed.

(* The two together: what the refinement goes by. *)
Definition held_inv (s : state) : Prop :=
  groups_inv s ∧ classes_agree held (state_statics s).

Lemma held_inv_kept : kept_by_step held_inv.
Proof.
  intros s s' [Hterms Hagree] Hstep.
  split; [ exact (groups_inv_step s s' Hterms Hstep) | ].
  destruct (step_statics s s' Hstep)
    as [-> | (bindings & body & Hexpr & Heval)];
    [ exact Hagree | ].
  destruct Hterms as (Hgroups & _). rewrite Hexpr in Hgroups.
  simpl in Hgroups. apply andb_true_iff in Hgroups as [Hgroups _].
  apply andb_true_iff in Hgroups as [Hgroup _].
  exact (group_keeps_classes held (state_vars s) bindings (state_statics s)
           (state_statics s') Hgroup Hagree Heval).
Qed.

Lemma held_inv_run (n : nat) :
  ∀ s s' : state, held_inv s -> run n s = Step s' -> held_inv s'.
Proof.
  induction n as [ | n IH]; intros s s' Hinv; simpl.
  - intros [= <-]. exact Hinv.
  - destruct (step s) as [next | k vs | ] eqn:Hstep; try discriminate.
    exact (IH next s' (held_inv_kept s next Hinv Hstep)).
Qed.

(* The invariant does not look at the expression of a state but to ask
   the check of it. *)
Lemma held_inv_expr (e e' : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  groups_hold held e' = groups_hold held e ->
  held_inv (State e vars conts statics hp rd)
  <-> held_inv (State e' vars conts statics hp rd).
Proof.
  intros Hsame. unfold held_inv, groups_inv, terms_inv. simpl_state.
  rewrite Hsame. reflexivity.
Qed.

End Held.

(* ======================================================================== *)
(* The reads made                                                           *)
(* ======================================================================== *)

(* A read made holds where its variable is bound to what a place of its
   class holds, of a symbol that the static environment binds. *)
Definition read_holds (held : held_places) (vars : env) (statics : static_env)
    (entry : place_class * variable) : Prop :=
  ∃ (s : symbol) (p : static_place)
    (allocated : statically_allocated_value) (v : value),
    (s, p, fst entry) ∈ held
    ∧ lookup s (static_values statics) = Some allocated
    ∧ place_value allocated p = Some v
    ∧ lookup (snd entry) vars = Some v.

Definition reads_hold (held : held_places) (made : reads_made) (vars : env)
    (statics : static_env) : Prop :=
  ∀ entry : place_class * variable,
    entry ∈ made -> read_holds held vars statics entry.

Lemma reads_hold_nil (held : held_places) (vars : env)
    (statics : static_env) :
  reads_hold held [] vars statics.
Proof. intros entry []. Qed.

Lemma reads_hold_forget (held : held_places) (made : reads_made) (vars : env)
    (statics : static_env) (x : variable) (v : value) :
  reads_hold held made vars statics ->
  reads_hold held (forget_read x made) ((x, v) :: vars) statics.
Proof.
  intros Hhold [c y] Hin. apply filter_In in Hin as [Hin Hother].
  apply negb_true_iff in Hother. cbn [snd] in Hother.
  destruct (Hhold (c, y) Hin)
    as (s & p & allocated & w & H1 & H2 & H3 & H4).
  exists s, p, allocated, w. repeat split; try assumption.
  cbn [snd lookup] in H4 |- *. rewrite (Pos.eqb_sym y x), Hother. exact H4.
Qed.

Lemma reads_hold_bind_params (held : held_places) (statics : static_env)
    (params : list variable) :
  ∀ (made : reads_made) (vs : list value) (vars handler_vars : env),
    reads_hold held made vars statics ->
    bind_params params vs vars = Some handler_vars ->
    reads_hold held (forget_reads params made) handler_vars statics.
Proof.
  induction params as [ | param params IH];
    intros made [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, reads_hold_forget, Hhold.
Qed.

(* A static environment that binds more. *)
Lemma reads_hold_extend (held : held_places) (made : reads_made) (vars : env)
    (statics statics' : static_env) :
  reads_hold held made vars statics ->
  statics_extend statics statics' ->
  reads_hold held made vars statics'.
Proof.
  intros Hhold [_ Hvalues] entry Hin.
  destruct (Hhold entry Hin) as (s & p & allocated & w & H1 & H2 & H3 & H4).
  exists s, p, allocated, w. repeat split; try assumption.
  apply Hvalues, H2.
Qed.

Lemma reads_hold_cons (held : held_places) (made : reads_made) (vars : env)
    (statics : static_env) (entry : place_class * variable) :
  read_holds held vars statics entry ->
  reads_hold held made vars statics ->
  reads_hold held (entry :: made) vars statics.
Proof.
  intros Hentry Hhold entry' [<- | Hin]; [ exact Hentry | exact (Hhold _ Hin) ].
Qed.

Theorem share_read_sound : share_read_sound_statement.
Proof.
  intros held statics vars d c s p allocated v w Hagree Hclass Hin Hs Hv Hd.
  unfold read_class in Hclass.
  destruct (read_of d) as [[s' p'] | ] eqn:Hread; [ | discriminate ].
  destruct (read_of_place statics vars d s' p' w Hread Hd)
    as (allocated' & Hs' & Hw).
  exact (Hagree s s' p p' c allocated allocated' v w Hin
           (class_of_in held s' p' c Hclass) Hs Hs' Hv Hw).
Qed.

(* ======================================================================== *)
(* What the rewrite keeps                                                   *)
(* ======================================================================== *)

Lemma share_reads_first_symbol (held : held_places)
    (bindings : list static_binding) :
  first_symbol (map (share_reads_in_binding held) bindings)
  = first_symbol bindings.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    cbn [map first_symbol share_reads_in_binding];
    try destruct (code_is_stub c); cbn [first_symbol]; congruence.
Qed.

Lemma share_reads_group_const (held : held_places) (s : symbol)
    (bindings : list static_binding) :
  group_const s (map (share_reads_in_binding held) bindings)
  = group_const s bindings.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    cbn [map group_const share_reads_in_binding];
    try destruct (code_is_stub c); cbn [group_const]; rewrite ?IH;
    reflexivity.
Qed.

Lemma share_reads_group_holds (held held' : held_places)
    (bindings : list static_binding) :
  group_holds held' (map (share_reads_in_binding held) bindings)
  = group_holds held' bindings.
Proof.
  unfold group_holds.
  induction held' as [ | [[s p] [first x]] held' IH]; cbn [forallb];
    [ reflexivity | ].
  rewrite IH. unfold entry_meets_group, entry_of_group.
  rewrite share_reads_first_symbol, share_reads_group_const. reflexivity.
Qed.

(* The check of a group does not look at its code. *)
Lemma first_symbol_code (before after : list static_binding) (id : code_id)
    (c c' : code) :
  first_symbol (before ++ Bind_code id c :: after)
  = first_symbol (before ++ Bind_code id c' :: after).
Proof.
  induction before as [ | [id0 c0 | id0 | sym0 const0] before IH];
    cbn [app first_symbol]; congruence.
Qed.

Lemma group_const_code (s : symbol) (before after : list static_binding)
    (id : code_id) (c c' : code) :
  group_const s (before ++ Bind_code id c :: after)
  = group_const s (before ++ Bind_code id c' :: after).
Proof.
  induction before as [ | [id0 c0 | id0 | sym0 const0] before IH];
    cbn [app group_const]; rewrite ?IH; reflexivity.
Qed.

Lemma group_holds_code (held : held_places)
    (before after : list static_binding) (id : code_id) (c c' : code) :
  group_holds held (before ++ Bind_code id c :: after)
  = group_holds held (before ++ Bind_code id c' :: after).
Proof.
  unfold group_holds.
  induction held as [ | [[s p] [first x]] held IH]; cbn [forallb];
    [ reflexivity | ].
  rewrite IH. unfold entry_meets_group, entry_of_group.
  rewrite (first_symbol_code before after id c c'),
    (group_const_code s before after id c c').
  reflexivity.
Qed.

(* The rewrite changes no static constant and makes no group. *)
Fixpoint share_reads_groups_hold_expr (held held' : held_places) (e : expr)
    {struct e} :
  ∀ made : reads_made,
    groups_hold held' (share_reads held made e) = groups_hold held' e
with share_reads_groups_hold_handler (held held' : held_places)
    (h : continuation_handler) {struct h} :
  ∀ made : reads_made,
    handler_groups_hold held' (share_reads_in_handler held made h)
    = handler_groups_hold held' h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros made; cbn [share_reads groups_hold]; try reflexivity.
    + destruct (read_class held d) as [c | ];
        [ destruct (read_before made c) | ];
        cbn [groups_hold]; apply share_reads_groups_hold_expr.
    + rewrite (share_reads_groups_hold_expr held held' body),
        share_reads_group_holds.
      f_equal. f_equal.
      induction bindings
        as [ | [id [params my_closure ret exn metadata code_body] | id
               | sym const]
               bindings IH];
        cbn [map forallb]; [ reflexivity | | | ].
      * cbn [share_reads_in_binding code_is_stub].
        destruct (cm_is_stub metadata); [ rewrite IH; reflexivity | ].
        cbn [share_reads_in_code code_of_binding_groups_hold
               code_groups_hold].
        rewrite (share_reads_groups_hold_expr held held' code_body), IH.
        reflexivity.
      * cbn [share_reads_in_binding]. rewrite IH. reflexivity.
      * cbn [share_reads_in_binding]. rewrite IH. reflexivity.
    + rewrite (share_reads_groups_hold_handler held held' h),
        (share_reads_groups_hold_expr held held' body).
      reflexivity.
    + rewrite (share_reads_groups_hold_expr held held' body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (share_reads_groups_hold_handler held held' h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros made.
    cbn [share_reads_in_handler handler_groups_hold].
    apply share_reads_groups_hold_expr.
Qed.

Theorem share_reads_groups_hold : share_reads_groups_hold_statement.
Proof.
  intros held held' made e. apply share_reads_groups_hold_expr.
Qed.

Lemma share_reads_bindings_groups_hold (held held' : held_places)
    (bindings : list static_binding) :
  forallb (code_of_binding_groups_hold held')
    (map (share_reads_in_binding held) bindings)
  = forallb (code_of_binding_groups_hold held') bindings.
Proof.
  induction bindings
    as [ | [id [params my_closure ret exn metadata code_body] | id
           | sym const]
           bindings IH];
    cbn [map forallb]; [ reflexivity | | | ].
  - cbn [share_reads_in_binding code_is_stub].
    destruct (cm_is_stub metadata); [ rewrite IH; reflexivity | ].
    cbn [share_reads_in_code code_of_binding_groups_hold code_groups_hold].
    rewrite share_reads_groups_hold_expr, IH. reflexivity.
  - cbn [share_reads_in_binding]. rewrite IH. reflexivity.
  - cbn [share_reads_in_binding]. rewrite IH. reflexivity.
Qed.

(* The rewrite binds what the term binds. *)
Fixpoint share_reads_binds_statics (held : held_places) (e : expr) :
  ∀ made : reads_made,
    binds_statics (share_reads held made e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros made; cbn [share_reads binds_statics]; try reflexivity.
  - destruct (read_class held d) as [c | ];
      [ destruct (read_before made c) | ];
      cbn [binds_statics]; apply share_reads_binds_statics.
  - rewrite (share_reads_binds_statics held body). f_equal.
    induction bindings as [ | [id c | id | sym const] bindings IH];
      cbn [map existsb share_reads_in_binding];
      try destruct (code_is_stub c); cbn [binding_binds]; congruence.
  - destruct h as [params metadata handler].
    cbn [share_reads_in_handler handler_binds_statics].
    rewrite !share_reads_binds_statics. reflexivity.
  - rewrite (share_reads_binds_statics held body). f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      cbn [map existsb share_reads_in_handler handler_binds_statics];
      [ reflexivity | ].
    rewrite share_reads_binds_statics, IH. reflexivity.
Qed.

(* ======================================================================== *)
(* Refinement                                                               *)
(* ======================================================================== *)

Section Refines.

Variable held : held_places.

Let inv : state -> Prop := held_inv held.

Let Hkept : kept_by_step inv := held_inv_kept held.

(* The invariant with the reads made where a handler is defined: they
   hold in every static environment that binds more, so a step keeps
   the two together, and they are what the handler has where it
   runs. *)
Definition inv_where (made : reads_made) (vars : env) (s : state) : Prop :=
  inv s ∧ reads_hold held made vars (state_statics s).

Lemma inv_where_kept (made : reads_made) (vars : env) :
  kept_by_step (inv_where made vars).
Proof.
  intros s s' [Hinv Hreads] Hstep.
  split; [ exact (Hkept s s' Hinv Hstep) | ].
  exact (reads_hold_extend held made vars _ _ Hreads
           (step_extends_statics s s' Hstep)).
Qed.

Lemma inv_where_expr (made : reads_made) (def_vars : env) (e e' : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  groups_hold held e' = groups_hold held e ->
  inv_where made def_vars (State e vars conts statics hp rd)
  <-> inv_where made def_vars (State e' vars conts statics hp rd).
Proof.
  intros Hsame. unfold inv_where, inv. simpl_state.
  pose proof (held_inv_expr held e e' vars conts statics hp rd Hsame) as Hiff.
  tauto.
Qed.

(* Down the term. The code of a group and the handlers are rewritten
   one at a time
   (FlambdaContextWhereSpec.let_static_code_refines_from_statement and
   the like), each from the states that have the invariant; a handler,
   from those that have it with the reads made where it is defined. *)
Fixpoint share_reads_refines_aux (e : expr) {struct e} :
  ∀ (made : reads_made) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    inv (State e vars conts statics hp rd) ->
    reads_hold held made vars statics ->
    refines (State e vars conts statics hp rd)
      (State (share_reads held made e) vars conts statics hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros made vars conts statics hp rd Hinv Hreads; cbn [share_reads];
    try apply refines_refl.
  - (* Let *)
    destruct (exec_named statics hp rd vars d) as [[[w hp'] rd'] | ] eqn:Hd;
      [ | apply step_undefined_refines; rewrite step_let_exec, Hd;
          reflexivity ].
    destruct (kind_eqb (kind_of_value w) kind) eqn:Hkind;
      [ | apply step_undefined_refines; rewrite step_let_exec, Hd, Hkind;
          reflexivity ].
    assert (Hnext : inv (State body ((x, w) :: vars) conts statics hp' rd')).
    { apply (Hkept _ _ Hinv). rewrite step_let_exec, Hd, Hkind.
      reflexivity. }
    destruct (read_class held d) as [c | ] eqn:Hclass.
    + (* A read. *)
      unfold read_class in Hclass.
      destruct (read_of d) as [[s p] | ] eqn:Hread; [ | discriminate ].
      pose proof (exec_named_keeps_reads statics hp hp' rd rd' vars d w Hd)
        as ->.
      destruct (read_of_exec statics hp hp' rd rd vars d s p w Hread Hd)
        as [Heval ->].
      destruct (read_of_place statics vars d s p w Hread Heval)
        as (allocated & Hs & Hplace).
      pose proof (class_of_in held s p c Hclass) as Hin.
      destruct (read_before made c) as [v | ] eqn:Hbefore.
      * (* Of a class that has been read: the variable of that read
           holds the value. *)
        destruct (Hreads (c, v) (read_before_in made c v Hbefore))
          as (s1 & p1 & a1 & u & Hin1 & Hs1 & Hplace1 & Hv).
        cbn [fst snd] in Hin1, Hv.
        assert (Hsame : w = u).
        { exact (proj2 Hinv s1 s p1 p c a1 allocated u w Hin1 Hin Hs1 Hs
                   Hplace1 Hplace). }
        subst u.
        eapply refines_step;
          [ rewrite step_let_exec, Hd, Hkind; reflexivity
          | rewrite step_let_exec,
              (exec_named_not_prim statics hp rd vars (Simple (Var v)))
              by (intros q args; discriminate);
            cbn [eval_named eval_simple]; rewrite Hv; cbv beta iota;
            rewrite Hkind; reflexivity
          | ].
        apply share_reads_refines_aux;
          [ exact Hnext | apply reads_hold_forget, Hreads ].
      * (* Of a class that has not: the read is remembered. *)
        eapply refines_step;
          [ rewrite step_let_exec, Hd, Hkind; reflexivity
          | rewrite step_let_exec, Hd, Hkind; reflexivity
          | ].
        apply share_reads_refines_aux; [ exact Hnext | ].
        apply reads_hold_cons; [ | apply reads_hold_forget, Hreads ].
        exists s, p, allocated, w. cbn [fst snd lookup].
        rewrite Pos.eqb_refl. repeat split; assumption.
    + eapply refines_step;
        [ rewrite step_let_exec, Hd, Hkind; reflexivity
        | rewrite step_let_exec, Hd, Hkind; reflexivity
        | ].
      apply share_reads_refines_aux;
        [ exact Hnext | apply reads_hold_forget, Hreads ].
  - (* Let_static *)
    apply (refines_trans _
      (State (Let_static (map (share_reads_in_binding held) bindings) body)
         vars conts statics hp rd)).
    + assert (Hcode :
        ∀ before : list static_binding,
          inv
            (State (Let_static (before ++ bindings) body) vars conts statics
               hp rd)
          ->
          refines
            (State (Let_static (before ++ bindings) body) vars conts statics
               hp rd)
            (State
               (Let_static
                  (before ++ map (share_reads_in_binding held) bindings) body)
               vars conts statics hp rd)).
      { clear Hinv.
        induction bindings
          as [ | [id [params my_closure ret exn metadata code_body] | id
                 | sym const]
                 bindings IH];
          intros before Hbefore; cbn [map]; [ apply refines_refl | | | ].
        - cbn [share_reads_in_binding code_is_stub].
          destruct (cm_is_stub metadata).
          { (* The code of a stub is left. *)
            specialize
              (IH
                 (before
                  ++ [Bind_code id
                        (Code params my_closure ret exn metadata code_body)])).
            rewrite <- !app_assoc in IH. cbn [app] in IH.
            exact (IH Hbefore). }
          cbn [share_reads_in_code].
          assert (Hbefore' :
            inv
              (State
                 (Let_static
                    (before
                     ++ Bind_code id
                          (Code params my_closure ret exn metadata
                             (share_reads held [] code_body))
                     :: bindings)
                    body)
                 vars conts statics hp rd)).
          { revert Hbefore. apply held_inv_expr. cbn [groups_hold].
            rewrite !forallb_app.
            cbn [forallb code_of_binding_groups_hold code_groups_hold].
            rewrite share_reads_groups_hold_expr,
              (group_holds_code held before bindings id
                 (Code params my_closure ret exn metadata
                    (share_reads held [] code_body))
                 (Code params my_closure ret exn metadata code_body)).
            reflexivity. }
          apply (refines_trans _
            (State
               (Let_static
                  (before
                   ++ Bind_code id
                        (Code params my_closure ret exn metadata
                           (share_reads held [] code_body))
                   :: bindings)
                  body)
               vars conts statics hp rd)).
          + apply (let_static_code_refines_from inv before id params
                     my_closure ret exn metadata code_body
                     (share_reads held [] code_body) bindings body vars conts
                     statics hp rd Hkept Hbefore Hbefore').
            * symmetry. apply share_reads_binds_statics.
            * intros vs callee callee_vars return_to exn_to statics' hp' rd' _ _
                _.
              split.
              -- apply held_inv_expr, share_reads_groups_hold_expr.
              -- intros Hentry. apply share_reads_refines_aux;
                   [ exact Hentry | apply reads_hold_nil ].
          + specialize
              (IH
                 (before
                  ++ [Bind_code id
                        (Code params my_closure ret exn metadata
                           (share_reads held [] code_body))])).
            rewrite <- !app_assoc in IH. cbn [app] in IH.
            exact (IH Hbefore').
        - cbn [share_reads_in_binding].
          specialize (IH (before ++ [Bind_deleted_code id])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hbefore).
        - cbn [share_reads_in_binding].
          specialize (IH (before ++ [Bind_symbol sym const])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hbefore). }
      exact (Hcode [] Hinv).
    + assert (Hinv' :
        inv
          (State
             (Let_static (map (share_reads_in_binding held) bindings) body)
             vars conts statics hp rd)).
      { revert Hinv. apply held_inv_expr. cbn [groups_hold].
        rewrite share_reads_group_holds, share_reads_bindings_groups_hold.
        reflexivity. }
      apply let_static_refines_step.
      destruct
        (eval_static_bindings vars
           (map (share_reads_in_binding held) bindings) statics)
        as [extended | ] eqn:Heval;
        [ | exact I ].
      apply share_reads_refines_aux.
      * apply (Hkept _ _ Hinv'). rewrite step_let_static, Heval. reflexivity.
      * exact (reads_hold_extend held made vars statics extended Hreads
                 (eval_static_bindings_extend vars _ statics extended Heval)).
  - (* Let_cont_non_recursive *)
    destruct h as [params metadata handler]. cbn [share_reads_in_handler].
    set (handler' := share_reads held (forget_reads params made) handler).
    assert (Hinv' :
      inv
        (State
           (Let_cont_non_recursive k (Handler params metadata handler') body)
           vars conts statics hp rd)).
    { revert Hinv. apply held_inv_expr.
      cbn [groups_hold handler_groups_hold]. unfold handler'.
      rewrite share_reads_groups_hold_expr. reflexivity. }
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k (Handler params metadata handler') body)
         vars conts statics hp rd)).
    + apply (let_cont_handler_refines_from (inv_where made vars) k params
               metadata handler handler' body vars conts statics hp rd
               (inv_where_kept made vars) (conj Hinv Hreads)
               (conj Hinv' Hreads)).
      intros vs handler_vars conts' statics' hp' rd' _ Hbind. split.
      * apply inv_where_expr. unfold handler'.
        apply share_reads_groups_hold_expr.
      * intros [Hentry Hreads']. simpl_state.
        apply share_reads_refines_aux; [ exact Hentry | ].
        exact (reads_hold_bind_params held statics' params made vs vars
                 handler_vars Hreads' Hbind).
    + apply let_cont_non_recursive_refines_step.
      apply share_reads_refines_aux; [ | exact Hreads ].
      apply (Hkept _ _ Hinv'). apply step_let_cont_non_recursive.
  - (* Let_cont_recursive *)
    set (share :=
           fun kh : continuation * continuation_handler =>
             let '(k, h) := kh in (k, share_reads_in_handler held made h)).
    apply (refines_trans _
      (State (Let_cont_recursive (map share handlers) body) vars conts
         statics hp rd)).
    + assert (Hhandlers :
        ∀ before : list (continuation * continuation_handler),
          inv_where made vars
            (State (Let_cont_recursive (before ++ handlers) body) vars conts
               statics hp rd)
          ->
          refines
            (State (Let_cont_recursive (before ++ handlers) body) vars conts
               statics hp rd)
            (State (Let_cont_recursive (before ++ map share handlers) body)
               vars conts statics hp rd)).
      { clear Hinv.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          intros before Hbefore; cbn [map]; [ apply refines_refl | ].
        unfold share at 1. cbn [share_reads_in_handler].
        set (handler' := share_reads held (forget_reads params made) handler).
        assert (Hbefore' :
          inv_where made vars
            (State
               (Let_cont_recursive
                  (before ++ (k, Handler params metadata handler') :: handlers)
                  body)
               vars conts statics hp rd)).
        { revert Hbefore. apply inv_where_expr. cbn [groups_hold].
          rewrite !forallb_app. cbn [forallb handler_groups_hold].
          unfold handler'. rewrite share_reads_groups_hold_expr.
          reflexivity. }
        apply (refines_trans _
          (State
             (Let_cont_recursive
                (before ++ (k, Handler params metadata handler') :: handlers)
                body)
             vars conts statics hp rd)).
        - apply (let_cont_recursive_handler_refines_from (inv_where made vars)
                   before k params metadata handler handler' handlers body
                   vars conts statics _ rd_ (inv_where_kept made vars) Hbefore
                   Hbefore').
          + symmetry. apply share_reads_binds_statics.
          + intros vs handler_vars conts' statics' hp' rd' _ Hbind. split.
            * apply inv_where_expr. unfold handler'.
              apply share_reads_groups_hold_expr.
            * intros [Hentry Hreads']. simpl_state.
              apply share_reads_refines_aux; [ exact Hentry | ].
              exact (reads_hold_bind_params held statics' params made vs vars
                       handler_vars Hreads' Hbind).
        - specialize
            (IH (before ++ [(k, Handler params metadata handler')])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hbefore'). }
      exact (Hhandlers [] (conj Hinv Hreads)).
    + assert (Hinv' :
        inv
          (State (Let_cont_recursive (map share handlers) body) vars conts
             statics hp rd)).
      { revert Hinv. apply held_inv_expr. cbn [groups_hold]. f_equal.
        clear. induction handlers as [ | [k h] handlers IH];
          cbn [map forallb]; [ reflexivity | ].
        unfold share at 1. rewrite share_reads_groups_hold_handler, IH.
        reflexivity. }
      apply let_cont_recursive_refines_step; [ reflexivity | ].
      intros Hbinds. apply share_reads_refines_aux; [ | exact Hreads ].
      apply (Hkept _ _ Hinv').
      rewrite step_let_cont_recursive, Hbinds. reflexivity.
Qed.

End Refines.

(* ======================================================================== *)
(* Where a run starts                                                       *)
(* ======================================================================== *)

Lemma classes_agree_empty (held : held_places) :
  classes_agree held empty_static_env.
Proof.
  intros s s' p p' c allocated allocated' v v' _ _ Hs. discriminate Hs.
Qed.

(* What the static environment of the imports binds a symbol to holds
   no value at any place. *)
Lemma classes_agree_imports (held : held_places)
    (imports : list (symbol * import_class)) :
  classes_agree held (imports_statics imports).
Proof.
  intros s s' p p' c allocated allocated' v v' _ _ Hs _ Hv.
  unfold imports_statics in Hs. cbn [static_values] in Hs.
  apply lookup_in, in_map_iff in Hs as ([s0 c0] & Heq & _).
  injection Heq as _ <-. destruct p; discriminate Hv.
Qed.

Lemma code_groups_hold_ok (held : held_places) (statics : static_env) :
  Forall (fun binding => code_groups_hold held (snd binding) = true)
    (static_code statics) ->
  code_terms_ok (code_groups_hold held) statics.
Proof.
  intros Hcode id c Hlookup. rewrite Forall_forall in Hcode.
  exact (Hcode (id, c) (lookup_in id c _ Hlookup)).
Qed.

Lemma held_inv_from (held : held_places) (e : expr) (statics : static_env) :
  groups_hold held e = true ->
  Forall (fun binding => code_groups_hold held (snd binding) = true)
    (static_code statics) ->
  classes_agree held statics ->
  held_inv held (State e [] [] statics empty_heap no_reads).
Proof.
  intros He Hcode Hagree. split; [ | exact Hagree ].
  unfold groups_inv, terms_inv. simpl_state.
  exact (conj He (conj I (code_groups_hold_ok held statics Hcode))).
Qed.

(* What may follow a unit passes the check, whatever is known: it has
   no group. *)
Fixpoint no_static_groups_hold (held : held_places) (e : expr) {struct e} :
  binds_no_static e = true -> groups_hold held e = true.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros Hno; cbn [groups_hold]; try reflexivity.
  - exact (no_static_groups_hold held body Hno).
  - discriminate Hno.
  - destruct h as [params metadata handler].
    cbn [binds_no_static handler_binds_no_static] in Hno.
    apply andb_true_iff in Hno as [Hh Hb]. cbn [handler_groups_hold].
    rewrite (no_static_groups_hold held handler Hh),
      (no_static_groups_hold held body Hb).
    reflexivity.
  - cbn [binds_no_static] in Hno. apply andb_true_iff in Hno as [Hhs Hb].
    rewrite (no_static_groups_hold held body Hb), andb_true_r.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      cbn [forallb] in *; [ reflexivity | ].
    apply andb_true_iff in Hhs as [Hh Hrest].
    cbn [handler_binds_no_static] in Hh. cbn [handler_groups_hold].
    rewrite (no_static_groups_hold held handler Hh). exact (IH Hrest).
Qed.

Lemma observers_groups_hold (held : held_places) (slots : list value_slot)
    (conts : cont_env) :
  observers slots conts ->
  conts_terms_ok (handler_groups_hold held) conts.
Proof.
  unfold observers, conts_terms_ok.
  induction conts as [ | [k c] conts IH]; simpl;
    [ intros _; exact I | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  split; [ | exact (IH Hrest) ].
  revert Hc.
  destruct c
    as [ [params metadata body] vars kept recursive | k0 | c0 extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction; [ | intros _; exact I ].
  destruct vars; [ | contradiction ].
  destruct kept; [ | contradiction ].
  destruct recursive; [ | contradiction ].
  intros (Hno & _ & _).
  split; [ exact (no_static_groups_hold held body Hno) | ].
  split; [ reflexivity | exact I ].
Qed.

Lemma held_inv_observed (held : held_places) (slots : list value_slot)
    (e : expr) (conts : cont_env) (statics : static_env) :
  groups_hold held e = true ->
  Forall (fun binding => code_groups_hold held (snd binding) = true)
    (static_code statics) ->
  classes_agree held statics ->
  observers slots conts ->
  held_inv held (State e [] conts statics empty_heap no_reads).
Proof.
  intros He Hcode Hagree Hobservers. split; [ | exact Hagree ].
  unfold groups_inv, terms_inv. simpl_state.
  exact
    (conj He
       (conj (observers_groups_hold held slots conts Hobservers)
          (code_groups_hold_ok held statics Hcode))).
Qed.

Theorem classes_agree_kept : classes_agree_kept_statement.
Proof.
  intros held e n s He Hrun.
  exact
    (proj2
       (held_inv_run held n _ s
          (held_inv_from held e empty_static_env He (Forall_nil _)
             (classes_agree_empty held))
          Hrun)).
Qed.

(* ======================================================================== *)
(* The statements of refinement                                             *)
(* ======================================================================== *)

Theorem share_reads_refines_from : share_reads_refines_from_statement.
Proof.
  intros held e statics He Hcode Hagree.
  apply (share_reads_refines_aux held);
    [ exact (held_inv_from held e statics He Hcode Hagree)
    | apply reads_hold_nil ].
Qed.

Theorem share_reads_program_refines : share_reads_program_refines_statement.
Proof.
  intros held e He. unfold program_refines.
  exact (share_reads_refines_from held e empty_static_env He (Forall_nil _)
           (classes_agree_empty held)).
Qed.

Theorem share_static_reads_program_refines :
  share_static_reads_program_refines_statement.
Proof.
  intros e. unfold share_static_reads. cbv zeta.
  destruct (groups_hold (places_held e) e) eqn:He;
    [ | apply refines_refl ].
  exact (share_reads_program_refines _ e He).
Qed.

Theorem share_static_reads_refines_imports :
  share_static_reads_refines_imports_statement.
Proof.
  intros imports e. unfold unit_refines, share_static_reads. cbv zeta.
  destruct (groups_hold (places_held e) e) eqn:He;
    [ | apply refines_refl ].
  apply (share_reads_refines_from _ e (imports_statics imports) He);
    [ apply Forall_nil | apply classes_agree_imports ].
Qed.

Theorem share_reads_refines_observed : share_reads_refines_observed_statement.
Proof.
  intros held e conts statics He Hcode Hagree Hobservers.
  apply (share_reads_refines_aux held);
    [ exact (held_inv_observed held [] e conts statics He Hcode Hagree
               Hobservers)
    | apply reads_hold_nil ].
Qed.

Theorem share_static_reads_refines_observed :
  share_static_reads_refines_observed_statement.
Proof.
  intros imports conts e Hobservers. unfold share_static_reads. cbv zeta.
  destruct (groups_hold (places_held e) e) eqn:He;
    [ | apply refines_refl ].
  apply (share_reads_refines_observed _ e conts (imports_statics imports) He);
    [ apply Forall_nil | apply classes_agree_imports | exact Hobservers ].
Qed.

(* ======================================================================== *)
(* Row 4                                                                    *)
(* ======================================================================== *)

(* The rewrite changes the body of code only, so what a binding binds,
   by a function that does not look into code, is what it bound. *)
Section Names_kept.

Variable names_of : static_binding -> list positive.
Hypothesis names_of_code :
  ∀ (id : code_id) (c c' : code),
    names_of (Bind_code id c') = names_of (Bind_code id c).

Lemma share_reads_binding_names (held : held_places)
    (binding : static_binding) :
  names_of (share_reads_in_binding held binding) = names_of binding.
Proof.
  destruct binding as [id c | id | sym const]; cbn [share_reads_in_binding];
    [ destruct (code_is_stub c); [ reflexivity | apply names_of_code ]
    | reflexivity | reflexivity ].
Qed.

Lemma share_reads_bindings_names (held : held_places)
    (bindings : list static_binding) :
  flat_map names_of (map (share_reads_in_binding held) bindings)
  = flat_map names_of bindings.
Proof.
  induction bindings as [ | binding bindings IH]; cbn [map flat_map];
    [ reflexivity | ].
  rewrite share_reads_binding_names, IH. reflexivity.
Qed.

Fixpoint share_reads_static_names (held : held_places) (e : expr)
    {struct e} :
  ∀ made : reads_made,
    static_names names_of (share_reads held made e) = static_names names_of e
with share_reads_handler_static_names (held : held_places)
    (h : continuation_handler) {struct h} :
  ∀ made : reads_made,
    handler_static_names names_of (share_reads_in_handler held made h)
    = handler_static_names names_of h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros made; cbn [share_reads static_names]; try reflexivity.
    + destruct (read_class held d) as [c | ];
        [ destruct (read_before made c) | ];
        cbn [static_names]; apply share_reads_static_names.
    + rewrite share_reads_bindings_names,
        (share_reads_static_names held body).
      reflexivity.
    + rewrite (share_reads_handler_static_names held h),
        (share_reads_static_names held body).
      reflexivity.
    + rewrite (share_reads_static_names held body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite (share_reads_handler_static_names held h), IH. reflexivity.
  - destruct h as [params metadata handler]. intros made.
    cbn [share_reads_in_handler handler_static_names].
    apply share_reads_static_names.
Qed.

End Names_kept.

Lemma share_reads_group_consts (held : held_places) (all : list symbol)
    (ids : list code_id) (bindings : list static_binding) :
  ∀ before : list symbol,
    group_consts_in_scope before all ids
      (map (share_reads_in_binding held) bindings)
    = group_consts_in_scope before all ids bindings.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros before; cbn [map share_reads_in_binding];
    try destruct (code_is_stub c); cbn [group_consts_in_scope];
    [ reflexivity | apply IH | apply IH | apply IH | ].
  destruct (is_closure_const const); rewrite IH; reflexivity.
Qed.

Fixpoint share_reads_in_scope_expr (held : held_places) (e : expr)
    {struct e} :
  ∀ (made : reads_made) (syms : list symbol) (ids : list code_id),
    expr_statics_in_scope syms ids e = true ->
    expr_statics_in_scope syms ids (share_reads held made e) = true
with share_reads_in_scope_handler (held : held_places)
    (h : continuation_handler) {struct h} :
  ∀ (made : reads_made) (syms : list symbol) (ids : list code_id),
    handler_statics_in_scope syms ids h = true ->
    handler_statics_in_scope syms ids (share_reads_in_handler held made h)
    = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros made syms ids He; cbn [share_reads]; try exact He.
    + (* Let *)
      cbn [expr_statics_in_scope] in He.
      apply andb_true_iff in He as [Hd Hbody].
      destruct (read_class held d) as [c | ];
        [ destruct (read_before made c) as [v | ] | ];
        cbn [expr_statics_in_scope named_statics_in_scope
               simple_statics_in_scope];
        rewrite ?Hd;
        exact (share_reads_in_scope_expr held body _ syms ids Hbody).
    + (* Let_static *)
      assert (Hbindings :
        ∀ (all : list symbol) (ids' : list code_id),
          forallb (binding_statics_in_scope all ids') bindings = true ->
          forallb (binding_statics_in_scope all ids')
            (map (share_reads_in_binding held) bindings)
          = true).
      { clear He. intros all ids'.
        induction bindings
          as [ | [id [params my_closure ret exn metadata code_body] | id
                 | sym const]
                 bindings IH];
          cbn [map forallb]; [ reflexivity | | | ]; intros Hcode;
          apply andb_true_iff in Hcode as [Hc Hrest];
          apply andb_true_iff; (split; [ | exact (IH Hrest) ]).
        - cbn [share_reads_in_binding code_is_stub].
          destruct (cm_is_stub metadata); [ exact Hc | ].
          cbn [share_reads_in_code binding_statics_in_scope
                 code_statics_in_scope] in Hc |- *.
          exact (share_reads_in_scope_expr held code_body [] all ids' Hc).
        - reflexivity.
        - reflexivity. }
      cbn [expr_statics_in_scope] in He |- *. cbv zeta in He |- *.
      rewrite
        (share_reads_bindings_names symbols_of_binding
           (fun _ _ _ => eq_refl)),
        (share_reads_bindings_names declared_code_ids_of_binding
           (fun _ _ _ => eq_refl)),
        (share_reads_bindings_names closure_symbols_of_binding
           (fun _ _ _ => eq_refl)),
        share_reads_group_consts.
      apply andb_true_iff in He as [He Hbody].
      apply andb_true_iff in He as [Hgroup Hcode].
      apply andb_true_iff. split; [ apply andb_true_iff; split | ].
      * exact Hgroup.
      * exact (Hbindings _ _ Hcode).
      * exact (share_reads_in_scope_expr held body made _ _ Hbody).
    + (* Let_cont_non_recursive *)
      cbn [expr_statics_in_scope] in He |- *.
      apply andb_true_iff in He as [Hh Hbody].
      rewrite
        (share_reads_static_names symbols_of_binding (fun _ _ _ => eq_refl)),
        (share_reads_static_names declared_code_ids_of_binding
           (fun _ _ _ => eq_refl)).
      apply andb_true_iff. split.
      * exact (share_reads_in_scope_handler held h made _ _ Hh).
      * exact (share_reads_in_scope_expr held body made syms ids Hbody).
    + (* Let_cont_recursive *)
      assert (Hrec :
        ∀ (all : list symbol) (ids' : list code_id),
          forallb (fun '(_, h) => handler_statics_in_scope all ids' h)
            handlers
          = true ->
          forallb (fun '(_, h) => handler_statics_in_scope all ids' h)
            (map (fun '(k, h) => (k, share_reads_in_handler held made h))
               handlers)
          = true).
      { clear He. intros all ids'.
        induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
          [ reflexivity | ]; intros Hhandlers.
        apply andb_true_iff in Hhandlers as [Hh Hrest].
        apply andb_true_iff. split; [ | exact (IH Hrest) ].
        exact (share_reads_in_scope_handler held h made all ids' Hh). }
      cbn [expr_statics_in_scope] in He |- *.
      apply andb_true_iff in He as [Hhandlers Hbody].
      rewrite
        (share_reads_static_names symbols_of_binding (fun _ _ _ => eq_refl)),
        (share_reads_static_names declared_code_ids_of_binding
           (fun _ _ _ => eq_refl)).
      apply andb_true_iff. split.
      * exact (Hrec _ _ Hhandlers).
      * exact (share_reads_in_scope_expr held body made syms ids Hbody).
  - destruct h as [params metadata handler]. intros made syms ids Hh.
    cbn [share_reads_in_handler handler_statics_in_scope] in Hh |- *.
    exact (share_reads_in_scope_expr held handler _ syms ids Hh).
Qed.

Theorem share_reads_in_scope : share_reads_in_scope_statement.
Proof.
  intros held made syms ids e He.
  exact (share_reads_in_scope_expr held e made syms ids He).
Qed.

Theorem share_static_reads_in_scope : share_static_reads_in_scope_statement.
Proof.
  intros imports e He. unfold share_static_reads. cbv zeta.
  destruct (groups_hold (places_held e) e); [ | exact He ].
  unfold statics_in_scope in *.
  exact (share_reads_in_scope_expr _ e [] imports [] He).
Qed.

(* ======================================================================== *)
(* The free names                                                           *)
(* ======================================================================== *)

Lemma forget_read_vars (x : variable) (made : reads_made) :
  incl (map snd (forget_read x made)) (map snd made).
Proof.
  intros y Hy. apply in_map_iff in Hy as (entry & <- & Hin).
  apply filter_In in Hin as [Hin _]. apply in_map, Hin.
Qed.

Lemma forget_reads_vars (xs : list variable) :
  ∀ made : reads_made,
    incl (map snd (forget_reads xs made)) (map snd made).
Proof.
  induction xs as [ | x xs IH]; intros made; cbn [forget_reads];
    [ apply incl_refl | ].
  exact (incl_tran (IH _) (forget_read_vars x made)).
Qed.

(* Under a binder of [x], what was free in the body or among the names
   and [x] is free in the term or among the names. *)
Lemma incl_without_under (x : variable) (names free free' : list variable) :
  incl free' (free ++ x :: names) ->
  incl (without [x] free') (without [x] free ++ names).
Proof.
  intros Hincl y Hy. apply in_without in Hy as [Hy Hx].
  apply Hincl, in_app_or in Hy. apply in_or_app.
  destruct Hy as [Hy | [Heq | Hy]].
  - left. apply in_without. split; assumption.
  - destruct Hx. left. exact Heq.
  - right. exact Hy.
Qed.

Lemma incl_without_names (bound names free free' : list variable) :
  incl free' (free ++ names) ->
  incl (without bound free') (without bound free ++ names).
Proof.
  intros Hincl y Hy. apply in_without in Hy as [Hy Hbound].
  apply Hincl, in_app_or in Hy. apply in_or_app.
  destruct Hy as [Hy | Hy]; [ left | right; exact Hy ].
  apply in_without. split; assumption.
Qed.

(* A variable that is free in the term rewritten is free in the term,
   or is the variable of a read made. *)
Fixpoint share_reads_free_variables (held : held_places) (e : expr)
    {struct e} :
  ∀ (made : reads_made) (names : list variable),
    incl (map snd made) names ->
    incl (free_variables_of_expr (share_reads held made e))
      (free_variables_of_expr e ++ names)
with share_reads_handler_free_variables (held : held_places)
    (h : continuation_handler) {struct h} :
  ∀ (made : reads_made) (names : list variable),
    incl (map snd made) names ->
    incl (free_variables_of_handler (share_reads_in_handler held made h))
      (free_variables_of_handler h ++ names).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros made names Hmade; cbn [share_reads];
      try (apply incl_appl, incl_refl).
    + (* Let *)
      assert (Hforget : incl (map snd (forget_read x made)) (x :: names)).
      { exact (incl_tran (forget_read_vars x made) (incl_tl x Hmade)). }
      destruct (read_class held d) as [c | ];
        [ destruct (read_before made c) as [v | ] eqn:Hbefore | ];
        cbn [free_variables_of_expr].
      * intros y Hy. apply in_app_or in Hy. apply in_or_app.
        destruct Hy as [Hy | Hy].
        -- right. cbn [free_variables_of_named free_variables_of_simple] in Hy.
           destruct Hy as [<- | []]. apply Hmade.
           exact (in_map snd _ _ (read_before_in made c v Hbefore)).
        -- apply
             (incl_without_under x names _ _
                (share_reads_free_variables held body _ _ Hforget))
             in Hy.
           apply in_app_or in Hy.
           destruct Hy as [Hy | Hy];
             [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
      * intros y Hy. apply in_app_or in Hy. apply in_or_app.
        destruct Hy as [Hy | Hy];
          [ left; apply in_or_app; left; exact Hy | ].
        apply
          (incl_without_under x names _ _
             (share_reads_free_variables held body ((c, x) :: _) _
                (incl_cons (in_eq x names) Hforget)))
          in Hy.
        apply in_app_or in Hy.
        destruct Hy as [Hy | Hy];
          [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
      * intros y Hy. apply in_app_or in Hy. apply in_or_app.
        destruct Hy as [Hy | Hy];
          [ left; apply in_or_app; left; exact Hy | ].
        apply
          (incl_without_under x names _ _
             (share_reads_free_variables held body _ _ Hforget))
          in Hy.
        apply in_app_or in Hy.
        destruct Hy as [Hy | Hy];
          [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
    + (* Let_static *)
      cbn [free_variables_of_expr].
      assert (Hbindings :
        incl
          (flat_map free_variables_of_binding
             (map (share_reads_in_binding held) bindings))
          (flat_map free_variables_of_binding bindings)).
      { induction bindings
          as [ | [id [params my_closure ret exn metadata code_body] | id
                 | sym const]
                 bindings IH];
          cbn [map flat_map]; [ apply incl_refl | | | ];
          (apply incl_app_app; [ | exact IH ]); try apply incl_refl.
        cbn [share_reads_in_binding code_is_stub].
        destruct (cm_is_stub metadata); [ apply incl_refl | ].
        cbn [share_reads_in_code free_variables_of_binding
               free_variables_of_code].
        apply incl_without.
        pose proof
          (share_reads_free_variables held code_body [] []
             (incl_refl _))
          as Hbody.
        rewrite app_nil_r in Hbody. exact Hbody. }
      intros y Hy. apply in_app_or in Hy. apply in_or_app.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; left; exact (Hbindings y Hy) | ].
      apply (share_reads_free_variables held body made names Hmade),
        in_app_or in Hy.
      destruct Hy as [Hy | Hy];
        [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
    + (* Let_cont_non_recursive *)
      cbn [free_variables_of_expr].
      intros y Hy. apply in_app_or in Hy. apply in_or_app.
      destruct Hy as [Hy | Hy].
      * apply (share_reads_handler_free_variables held h made names Hmade),
          in_app_or in Hy.
        destruct Hy as [Hy | Hy];
          [ left; apply in_or_app; left; exact Hy | right; exact Hy ].
      * apply (share_reads_free_variables held body made names Hmade),
          in_app_or in Hy.
        destruct Hy as [Hy | Hy];
          [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
    + (* Let_cont_recursive *)
      cbn [free_variables_of_expr].
      assert (Hhandlers :
        incl
          (flat_map (fun '(_, h) => free_variables_of_handler h)
             (map (fun '(k, h) => (k, share_reads_in_handler held made h))
                handlers))
          (flat_map (fun '(_, h) => free_variables_of_handler h) handlers
           ++ names)).
      { induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
          [ intros y [] | ].
        intros y Hy. apply in_app_or in Hy. apply in_or_app.
        destruct Hy as [Hy | Hy].
        - apply (share_reads_handler_free_variables held h made names Hmade),
            in_app_or in Hy.
          destruct Hy as [Hy | Hy];
            [ left; apply in_or_app; left; exact Hy | right; exact Hy ].
        - apply IH, in_app_or in Hy.
          destruct Hy as [Hy | Hy];
            [ left; apply in_or_app; right; exact Hy | right; exact Hy ]. }
      intros y Hy. apply in_app_or in Hy. apply in_or_app.
      destruct Hy as [Hy | Hy].
      * apply Hhandlers, in_app_or in Hy.
        destruct Hy as [Hy | Hy];
          [ left; apply in_or_app; left; exact Hy | right; exact Hy ].
      * apply (share_reads_free_variables held body made names Hmade),
          in_app_or in Hy.
        destruct Hy as [Hy | Hy];
          [ left; apply in_or_app; right; exact Hy | right; exact Hy ].
  - destruct h as [params metadata handler]. intros made names Hmade.
    cbn [share_reads_in_handler free_variables_of_handler].
    apply incl_without_names.
    exact (share_reads_free_variables held handler _ names
             (incl_tran (forget_reads_vars params made) Hmade)).
Qed.

(* The rewrite names the continuations that the term names. *)
Fixpoint share_reads_free_continuations (held : held_places) (e : expr)
    {struct e} :
  ∀ made : reads_made,
    free_continuations_of_expr (share_reads held made e)
    = free_continuations_of_expr e
with share_reads_handler_free_continuations (held : held_places)
    (h : continuation_handler) {struct h} :
  ∀ made : reads_made,
    free_continuations_of_handler (share_reads_in_handler held made h)
    = free_continuations_of_handler h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros made; cbn [share_reads free_continuations_of_expr];
      try reflexivity.
    + destruct (read_class held d) as [c | ];
        [ destruct (read_before made c) | ];
        cbn [free_continuations_of_expr];
        apply share_reads_free_continuations.
    + rewrite (share_reads_free_continuations held body). f_equal.
      induction bindings
        as [ | [id [params my_closure ret exn metadata code_body] | id
               | sym const]
               bindings IH];
        cbn [map flat_map]; [ reflexivity | | | ]; rewrite IH;
        try reflexivity.
      cbn [share_reads_in_binding code_is_stub].
      destruct (cm_is_stub metadata); [ reflexivity | ].
      cbn [share_reads_in_code free_continuations_of_binding
             free_continuations_of_code].
      rewrite (share_reads_free_continuations held code_body). reflexivity.
    + rewrite (share_reads_handler_free_continuations held h),
        (share_reads_free_continuations held body).
      reflexivity.
    + rewrite (share_reads_free_continuations held body), map_map.
      f_equal; [ apply map_ext; intros [k h]; reflexivity | ]. f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite (share_reads_handler_free_continuations held h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros made.
    cbn [share_reads_in_handler free_continuations_of_handler].
    apply share_reads_free_continuations.
Qed.

(* The code of the term rewritten is closed, if that of the term is. *)
Fixpoint share_reads_code_closed (held : held_places) (e : expr) {struct e} :
  ∀ made : reads_made,
    code_closed e = true -> code_closed (share_reads held made e) = true
with share_reads_handler_code_closed (held : held_places)
    (h : continuation_handler) {struct h} :
  ∀ made : reads_made,
    handler_code_closed h = true ->
    handler_code_closed (share_reads_in_handler held made h) = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros made He; cbn [share_reads]; try exact He.
    + cbn [code_closed] in He.
      destruct (read_class held d) as [c | ];
        [ destruct (read_before made c) | ];
        cbn [code_closed]; exact (share_reads_code_closed held body _ He).
    + cbn [code_closed] in He |- *.
      apply andb_true_iff in He as [Hbindings Hbody].
      rewrite (share_reads_code_closed held body made Hbody), andb_true_r.
      induction bindings
        as [ | [id [params my_closure ret exn metadata code_body] | id
               | sym const]
               bindings IH];
        cbn [map forallb] in *; [ reflexivity | | | ];
        apply andb_true_iff in Hbindings as [Hc Hrest];
        rewrite (IH Hrest), andb_true_r; try reflexivity.
      cbn [share_reads_in_binding code_is_stub].
      destruct (cm_is_stub metadata); [ exact Hc | ].
      cbn [share_reads_in_code binding_code_closed code_is_closed
             free_variables_of_code free_continuations_of_code]
        in Hc |- *.
      apply andb_true_iff in Hc as [Hc Hclosed].
      apply andb_true_iff in Hc as [Hv Hk].
      rewrite (share_reads_free_continuations held code_body), Hk,
        (share_reads_code_closed held code_body [] Hclosed), !andb_true_r.
      refine (incl_is_empty _ _ (incl_without _ _ _ _) Hv).
      pose proof
        (share_reads_free_variables held code_body [] [] (incl_refl _))
        as Hfree.
      rewrite app_nil_r in Hfree. exact Hfree.
    + cbn [code_closed] in He |- *. apply andb_true_iff in He as [Hh Hbody].
      rewrite (share_reads_handler_code_closed held h made Hh),
        (share_reads_code_closed held body made Hbody).
      reflexivity.
    + cbn [code_closed] in He |- *.
      apply andb_true_iff in He as [Hhandlers Hbody].
      rewrite (share_reads_code_closed held body made Hbody), andb_true_r.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb] in *;
        [ reflexivity | ].
      apply andb_true_iff in Hhandlers as [Hh Hrest].
      rewrite (share_reads_handler_code_closed held h made Hh).
      exact (IH Hrest).
  - destruct h as [params metadata handler]. intros made Hh.
    cbn [share_reads_in_handler handler_code_closed] in Hh |- *.
    exact (share_reads_code_closed held handler _ Hh).
Qed.

Theorem share_static_reads_keeps_scope :
  share_static_reads_keeps_scope_statement.
Proof.
  intros e Hclosed. unfold share_static_reads. cbv zeta.
  destruct (groups_hold (places_held e) e);
    [ | split; [ exact Hclosed | split; intros name Hname; exact Hname ] ].
  split; [ exact (share_reads_code_closed _ e [] Hclosed) | ].
  split.
  - intros x Hx. apply among_In. apply among_In in Hx.
    pose proof
      (share_reads_free_variables (places_held e) e [] [] (incl_refl _) x Hx)
      as Hfree.
    rewrite app_nil_r in Hfree. exact Hfree.
  - intros k Hk. rewrite share_reads_free_continuations in Hk. exact Hk.
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The two statements are proved from the definitions by the tactic of
   FlambdaEquivarianceTactic.v ([equivariance]), which states and
   proves by itself the laws of the functions that the rewrite and
   its check are made of. What it is given:
   - how what is known, the reads made and a class are renamed: their
     types are lists and pairs of positives, and do not say it;
   - that what is known, renamed, goes into an append with its name,
     which is not unfolded where it is at something that is not a
     list written out; and that [is_some] does not look at what an
     option holds;
   - five laws of functions of the pass, which it proves, so that it
     has them before the functions that call them are unfolded. That
     of [same_class] is at the class of an entry of the reads made,
     in the form that computation gives where they are renamed: a
     pair. *)
Definition rename_class (r : renaming) (c : place_class) : place_class :=
  (fst c, rename_variable r (snd c)).

#[local] Instance held_renaming : Renaming_of held_places := rename_held.
#[local] Instance reads_made_renaming : Renaming_of reads_made :=
  rename_reads_made.
#[local] Instance class_renaming : Renaming_of place_class := rename_class.

Lemma rename_held_app (r : renaming) (held held' : held_places) :
  rename_held r (held ++ held') = rename_held r held ++ rename_held r held'.
Proof. apply map_app. Qed.
#[local] Hint Rewrite rename_held_app : equivariance.
#[local] Arguments rename_held : simpl never.

Lemma is_some_option_map {A B : Type} (f : A -> B) (o : option A) :
  is_some (option_map f o) = is_some o.
Proof. destruct o; reflexivity. Qed.
#[local] Hint Rewrite @is_some_option_map : equivariance.

Lemma first_symbol_renamed (r : renaming) (bindings : list static_binding) :
  first_symbol (map (rename_static_binding r) bindings)
  = first_symbol bindings.
Proof. equivariance. Qed.
#[local] Hint Rewrite first_symbol_renamed : equivariance.

Lemma group_const_renamed (r : renaming) (s : symbol)
    (bindings : list static_binding) :
  group_const s (map (rename_static_binding r) bindings)
  = option_map (rename_static_const r) (group_const s bindings).
Proof. equivariance. Qed.
#[local] Hint Rewrite group_const_renamed : equivariance.

(* What is known of the term renamed is what is known of the term,
   renamed. Stated: with it the laws that follow pass with the one
   pass of the database, which is faster; it does not. *)
Lemma places_held_renamed (r : renaming) (e : expr) :
  places_held (rename_expr r e) = rename_held r (places_held e).
Proof. equivariance. Qed.
#[local] Hint Rewrite places_held_renamed : equivariance.

#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

Lemma same_class_of_entry_renamed (r : renaming) (c : place_class)
    (first : symbol) (x : variable) :
  same_class (rename_class r c) (first, rename_variable r x)
  = same_class c (first, x).
Proof. equivariance. Qed.
#[local] Hint Rewrite same_class_of_entry_renamed : equivariance.

Lemma forget_read_renamed (r : renaming) (x : variable) (made : reads_made) :
  forget_read (rename_variable r x) (rename_reads_made r made)
  = rename_reads_made r (forget_read x made).
Proof. equivariance. Qed.
#[local] Hint Rewrite forget_read_renamed : equivariance.

Lemma share_reads_renamed (r : renaming) (held : held_places) (e : expr) :
  ∀ made : reads_made,
    share_reads (rename_held r held) (rename_reads_made r made)
      (rename_expr r e)
    = rename_expr r (share_reads held made e).
Proof. equivariance. Qed.
#[local] Hint Rewrite share_reads_renamed : equivariance.

Theorem share_reads_equivariant : share_reads_equivariant_statement.
Proof. intros r held made e. apply share_reads_renamed. Qed.

Theorem share_static_reads_equivariant :
  share_static_reads_equivariant_statement.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* A store to a symbol                                                      *)
(* ======================================================================== *)

Theorem exec_prim_store_to_symbol : exec_prim_store_to_symbol_statement.
Proof.
  intros statics hp kind init field s new. unfold exec_prim.
  destruct (forallb (argument_bound statics hp) [Value_symbol s; new]);
    reflexivity.
Qed.

(* The machine runs a store by [exec_prim] on the values of its
   arguments, and the first is the symbol. *)
Theorem store_to_symbol_undefined : store_to_symbol_undefined_statement.
Proof.
  intros statics hp rd vars kind init field s new.
  rewrite exec_named_prim.
  destruct (eval_simples vars [Symbol s; new]) as [vs | ] eqn:Hvs;
    [ | reflexivity ].
  simpl in Hvs. destruct (eval_simple vars new) as [w | ]; [ | discriminate ].
  injection Hvs as <-. rewrite exec_prim_store_to_symbol. reflexivity.
Qed.

(* ======================================================================== *)
(* Eight units                                                              *)
(* ======================================================================== *)

Theorem share_static_reads_shares : share_static_reads_shares_statement.
Proof. reflexivity. Qed.

Theorem share_static_reads_two_variables :
  share_static_reads_two_variables_statement.
Proof. reflexivity. Qed.

Theorem share_static_reads_slot_twice :
  share_static_reads_slot_twice_statement.
Proof. split; reflexivity. Qed.

Theorem share_static_reads_two_groups :
  share_static_reads_two_groups_statement.
Proof. reflexivity. Qed.

Theorem share_static_reads_bound_again :
  share_static_reads_bound_again_statement.
Proof. split; reflexivity. Qed.

Theorem share_static_reads_in_stub : share_static_reads_in_stub_statement.
Proof. reflexivity. Qed.

Theorem share_static_reads_in_handler :
  share_static_reads_in_handler_statement.
Proof. split; reflexivity. Qed.

Theorem share_static_reads_store : share_static_reads_store_statement.
Proof. split; reflexivity. Qed.
