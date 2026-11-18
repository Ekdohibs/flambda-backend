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


(* The proofs of FlambdaReaperWrapperShapeSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionProofs
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperLeast FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperWrapperSpec FlambdaReaperWrapperShapeSpec.
From OxCamlFlambda2 Require FlambdaMapProofs FlambdaReaperFlowProofs
  FlambdaReaperLeastProofs FlambdaReaperWrapperProofs
  FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The invariant of the traversal                                           *)
(* ======================================================================== *)

Abbreviation cid := Field_code_id_of_call_witness.
Abbreviation env_conts := FlambdaReaperTraversal.env_conts.

Definition is_cid (n : node) : bool :=
  match n with
  | Node_code_id _ => true
  | _ => false
  end.

Definition is_cid_field (f : field) : bool :=
  match f with
  | Field_code_id_of_call_witness => true
  | _ => false
  end.

(* A code id is held only in the field of the code id, and is never
   aliased from, read off or given an argument. *)
Definition fact_ok (fct : fact) : bool :=
  match fct with
  | Fact_alias _ from | Fact_propagate _ _ from
  | Fact_alias_if_any_source _ _ from => negb (is_cid from)
  | Fact_accessor _ _ base => negb (is_cid base)
  | Fact_argument from _ base => negb (is_cid from) && negb (is_cid base)
  | Fact_constructor _ f from => negb (is_cid from) || is_cid_field f
  | Fact_use _ _ | Fact_parameter _ _ _ | Fact_any_usage _
  | Fact_any_source _ | Fact_any_source_if_used _ _ => true
  end.

(* [fact_ok], with no read of a code id and no argument. *)
Definition plain_fact (fct : fact) : bool :=
  fact_ok fct
  && match fct with
     | Fact_accessor _ f _ => negb (is_cid_field f)
     | Fact_argument _ _ _ => false
     | _ => true
     end.

Definition dep_nodes (cd : code_dep) : list node :=
  cd_my_closure cd :: cd_known_arity_call_witness cd
  :: cd_params cd ++ cd_unknown_arity_call_witnesses cd.

(* The facts are [fact_ok]; a code id is read only into a node with any
   usage; a node given an argument has its code id read; the nodes of
   the deps that facts alias from are not code ids. *)
Record cinv (a : acc) : Prop := {
  ci_ok : ∀ fct, fct ∈ acc_facts a -> fact_ok fct = true;
  ci_read : ∀ x b,
    Fact_accessor x cid b ∈ acc_facts a -> Fact_any_usage x ∈ acc_facts a;
  ci_arg : ∀ y i b,
    Fact_argument y i b ∈ acc_facts a ->
    ∃ x, Fact_accessor x cid b ∈ acc_facts a;
  ci_code_deps : ∀ id cd n,
    (id, cd) ∈ acc_code_deps a -> n ∈ dep_nodes cd -> is_cid n = false;
  ci_apply_deps : ∀ d,
    d ∈ acc_apply_deps a -> is_cid (ad_apply_closure d) = false;
  ci_closure_deps : ∀ d,
    d ∈ acc_closure_deps a ->
    is_cid (cld_let_bound_name_of_the_closure d) = false
}.

Lemma is_cid_simple (s : simple) : is_cid (node_of_simple s) = false.
Proof. destruct s; reflexivity. Qed.

Lemma cinv_same (a a' : acc) :
  acc_facts a' = acc_facts a -> acc_code_deps a' = acc_code_deps a ->
  acc_apply_deps a' = acc_apply_deps a ->
  acc_closure_deps a' = acc_closure_deps a -> cinv a -> cinv a'.
Proof.
  intros E1 E2 E3 E4 [H1 H2 H3 H4 H5 H6].
  constructor; rewrite ?E1, ?E2, ?E3, ?E4; assumption.
Qed.

Lemma cinv_add_facts (L : list fact) (a : acc) :
  cinv a ->
  (∀ fct, fct ∈ L -> fact_ok fct = true) ->
  (∀ x b, Fact_accessor x cid b ∈ L ->
     Fact_any_usage x ∈ L ++ acc_facts a) ->
  (∀ y i b, Fact_argument y i b ∈ L ->
     ∃ x, Fact_accessor x cid b ∈ L ++ acc_facts a) ->
  cinv (add_facts L a).
Proof.
  intros [H1 H2 H3 H4 H5 H6] HL1 HL2 HL3.
  constructor; cbn [add_facts acc_facts acc_code_deps acc_apply_deps
                    acc_closure_deps]; auto.
  - intros fct Hin. apply in_app_or in Hin as [Hin | Hin]; auto.
  - intros x b Hin. apply in_app_or in Hin as [Hin | Hin];
      [ exact (HL2 x b Hin) | ].
    apply in_or_app. right. exact (H2 x b Hin).
  - intros y i b Hin. apply in_app_or in Hin as [Hin | Hin];
      [ exact (HL3 y i b Hin) | ].
    destruct (H3 y i b Hin) as (x & Hx). exists x. apply in_or_app.
    right. exact Hx.
Qed.

Lemma cinv_add_plain (L : list fact) (a : acc) :
  cinv a -> (∀ fct, fct ∈ L -> plain_fact fct = true) ->
  cinv (add_facts L a).
Proof.
  intros Hinv HL. apply cinv_add_facts; [ exact Hinv | | | ].
  - intros fct Hin. pose proof (HL fct Hin) as H. unfold plain_fact in H.
    apply andb_true_iff in H as [H _]. exact H.
  - intros x b Hin. exfalso. pose proof (HL _ Hin) as H. revert H.
    cbv [plain_fact fact_ok is_cid_field negb andb orb].
    destruct (is_cid b); discriminate.
  - intros y i b Hin. exfalso. pose proof (HL _ Hin) as H. revert H.
    cbv [plain_fact fact_ok is_cid_field negb andb orb].
    destruct (is_cid y), (is_cid b); discriminate.
Qed.

(* Each fact of a list given by a term is [plain_fact]. *)
Ltac plain_tac :=
  let fct := fresh "fct" in let H := fresh "H" in
  intros fct H;
  repeat match type of H with
  | In _ (_ ++ _) => apply in_app_or in H; destruct H as [H | H]
  | In _ (map _ _) =>
    let y := fresh "y" in let Hy := fresh "Hy" in
    apply in_map_iff in H; destruct H as [y [H Hy]]
  | In _ (flat_map _ _) =>
    let y := fresh "y" in let Hy := fresh "Hy" in
    apply in_flat_map in H; destruct H as [y [Hy H]]
  | In _ (_ :: _) => destruct H as [H | H]
  | In _ [] => destruct H
  | In _ (match ?t with _ => _ end) => destruct t
  end;
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  subst; cbv [plain_fact fact_ok negb andb orb];
  rewrite ?is_cid_simple;
  repeat match goal with
  | H : is_cid ?n = false |- context [is_cid ?n] => rewrite H
  | H : is_cid_field ?n = false |- context [is_cid_field ?n] => rewrite H
  end;
  cbn [is_cid is_cid_field negb andb orb fst snd fresh];
  try reflexivity.

Lemma cinv_note_simple (s : simple) (a : acc) :
  cinv a -> cinv (note_simple s a).
Proof. destruct s; apply cinv_same; reflexivity. Qed.

Lemma cinv_note_simples (l : list simple) (a : acc) :
  cinv a -> cinv (note_simples l a).
Proof.
  destruct (FlambdaReaperLeastProofs.AutoWidgets.note_simples_fields l a)
    as (_ & H1 & H2 & H3 & H4).
  apply cinv_same; assumption.
Qed.

Lemma cinv_fresh (a : acc) : cinv a -> cinv (snd (fresh a)).
Proof. apply cinv_same; reflexivity. Qed.

Lemma cinv_fresh_list (n : nat) (a : acc) :
  cinv a -> cinv (snd (fresh_list n a)).
Proof.
  destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec n a)
    as (H1 & H2 & H3 & H4 & _).
  apply cinv_same; assumption.
Qed.

Lemma fresh_list_made (n : nat) (a : acc) (x : node) :
  x ∈ fst (fresh_list n a) -> is_cid x = false.
Proof.
  intros Hx.
  destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec n a)
    as (_ & _ & _ & _ & _ & H).
  destruct (H x Hx) as (m & -> & _). reflexivity.
Qed.

Lemma cinv_find_cont (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  cinv a -> cinv (snd (find_cont env k n a)).
Proof.
  intros Hinv. unfold find_cont.
  destruct (lookup k (env_conts env)); [ exact Hinv | ].
  pose proof (cinv_fresh_list n a Hinv) as H.
  destruct (fresh_list n a) as [params b]. cbn [snd] in *.
  apply cinv_add_plain; [ exact H | plain_tac ].
Qed.

Lemma find_cont_facts (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) (fct : fact) :
  fct ∈ acc_facts a -> fct ∈ acc_facts (snd (find_cont env k n a)).
Proof.
  intros H. unfold find_cont.
  destruct (lookup k (env_conts env)); [ exact H | ].
  destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec n a)
    as (H1 & _).
  destruct (fresh_list n a) as [params b]. cbn [snd add_facts acc_facts] in *.
  apply in_or_app. right. rewrite H1. exact H.
Qed.

Lemma cinv_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  cinv a -> cinv (cond_any_usage env s a).
Proof.
  intros Hinv. unfold cond_any_usage.
  pose proof (cinv_note_simple s a Hinv) as H.
  destruct (env_current_code_id env); apply cinv_add_plain; auto;
    plain_tac.
Qed.

Lemma cinv_cond_alias (env : traverse_env) (from to_ : node) (a : acc) :
  is_cid from = false -> cinv a -> cinv (cond_alias env from to_ a).
Proof.
  intros Hf Hinv. unfold cond_alias.
  destruct (env_current_code_id env); apply cinv_add_plain; auto;
    plain_tac.
Qed.

Lemma cinv_block_deps (base : node) (fields : list simple) (a : acc) :
  cinv a -> cinv (block_deps base fields a).
Proof.
  intros Hinv. unfold block_deps. apply cinv_add_plain;
    [ apply cinv_note_simples; exact Hinv | plain_tac ].
Qed.

Lemma cinv_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  cinv a -> cinv (traverse_prim env x p args a).
Proof.
  intros Hinv. unfold traverse_prim.
  assert (Hacc : ∀ f s b, is_cid_field f = false -> cinv b ->
             cinv (add_fact (Fact_accessor (Node_variable x) f
                               (node_of_simple s)) (note_simple s b))).
  { intros f s b Hf Hb. apply cinv_add_plain;
      [ apply cinv_note_simple; exact Hb | ].
    plain_tac. }
  assert (Hdefault :
    cinv
      (add_facts
         (Fact_any_source (Node_variable x)
          :: map (fun from => Fact_use (Node_variable x) from)
               (free_names_of_simples args))
         (if prim_has_arbitrary_effects p
          then cond_any_usage env (Var x) a else a))).
  { apply cinv_add_plain; [ | plain_tac ].
    destruct (prim_has_arbitrary_effects p);
      [ apply cinv_cond_any_usage; exact Hinv | exact Hinv ]. }
  destruct p as [ u | b | v | h | r ];
    [ destruct u | destruct b | destruct v | destruct h | destruct r ];
    repeat match goal with
    | |- cinv (match ?t with _ => _ end) => destruct t
    end;
    try exact Hdefault;
    try (apply cinv_block_deps; exact Hinv);
    try (apply Hacc; [ reflexivity | exact Hinv ]);
    try exact Hinv.
  - apply cinv_add_plain; [ | plain_tac ].
    apply Hacc; [ reflexivity | exact Hinv ].
  - apply cinv_add_plain; [ apply cinv_note_simple; exact Hinv | plain_tac ].
Qed.

Lemma cinv_fold {A : Type} (f : acc -> A -> acc) (l : list A) (a : acc) :
  (∀ y b, y ∈ l -> cinv b -> cinv (f b y)) ->
  cinv a -> cinv (List.fold_left f l a).
Proof.
  revert a. induction l as [ | y l IH]; intros a H Ha; cbn [List.fold_left];
    [ exact Ha | ].
  apply IH; [ intros y' b Hy' Hb; apply H; [ right; exact Hy' | exact Hb ]
            | apply H; [ left; reflexivity | exact Ha ] ].
Qed.

Lemma cinv_record_set (env : traverse_env)
    (closures : list (node * option code_id))
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) (a : acc) :
  (∀ base code, (base, code) ∈ closures -> is_cid base = false) ->
  (∀ f c, (f, c) ∈ function_slots -> is_cid c = false) ->
  cinv a ->
  cinv (record_set_of_closures_deps env closures function_slots value_slots
          a).
Proof.
  intros Hc Hf Hinv. unfold record_set_of_closures_deps.
  apply cinv_fold; [ | exact Hinv ].
  intros [base code] b Hin Hb. cbv beta iota.
  pose proof (Hc _ _ Hin) as Hbase.
  apply cinv_add_plain.
  - apply cinv_note_simples. destruct code as [id | ]; [ | exact Hb ].
    destruct Hb as [H1 H2 H3 H4 H5 H6]. constructor; cbn; auto.
    intros d [<- | Hd]; [ exact Hbase | auto ].
  - plain_tac. rewrite (Hf _ _ Hy). reflexivity.
Qed.

Lemma closures_of_set_spec (x : variable) (slot : function_slot)
    (others : list (function_slot * variable))
    (decls : list (function_slot * code_id)) (a : acc) :
  (cinv a -> cinv (snd (closures_of_set x slot others decls a)))
  ∧ ∀ f name id, (f, name, id) ∈ fst (closures_of_set x slot others decls a) ->
      is_cid name = false.
Proof.
  revert a. induction decls as [ | [f id] decls IH]; intros a;
    cbn [closures_of_set].
  - cbn [fst snd]. split; [ auto | intros f name id [] ].
  - match goal with
    | |- context [let '(_, _) := ?t in _] =>
      assert (Ht : (cinv a -> cinv (snd t)) ∧ is_cid (fst t) = false);
      [ | destruct t as [name b] eqn:E ]
    end.
    { destruct (Pos.eqb f slot); [ cbn; auto | ].
      destruct (lookup f others); [ cbn; auto | ].
      split; [ apply cinv_fresh | reflexivity ]. }
    cbn [fst snd] in Ht. destruct Ht as [H1 Hname].
    destruct (IH b) as [H2 H3].
    destruct (closures_of_set x slot others decls b) as [rest c] eqn:E2.
    cbn [fst snd] in *. split; [ auto | ].
    intros f' name' id' [Heq | Hin].
    + injection Heq as <- <- <-. exact Hname.
    + exact (H3 _ _ _ Hin).
Qed.

Lemma cinv_traverse_closure_of_set (env : traverse_env) (x : variable)
    (slot : function_slot) (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (a : acc) :
  cinv a ->
  cinv (traverse_closure_of_set env x slot function_decls value_slots a).
Proof.
  intros Hinv. unfold traverse_closure_of_set. cbv zeta.
  destruct (among slot _); [ | exact Hinv ].
  match goal with
  | |- context [closures_of_set ?x ?s ?o ?d a] =>
    destruct (closures_of_set_spec x s o d a) as [H1 H2];
    destruct (closures_of_set x s o d a) as [closures b] eqn:E
  end.
  cbn [fst snd] in *. apply cinv_record_set; [ | | exact (H1 Hinv) ].
  - intros base code Hin. apply in_map_iff in Hin.
    destruct Hin as [[[f name] id] [Heq Hin]].
    injection Heq as <- _. exact (H2 _ _ _ Hin).
  - intros f c Hin. apply in_map_iff in Hin.
    destruct Hin as [[[f' name] id] [Heq Hin]].
    injection Heq as _ <-. exact (H2 _ _ _ Hin).
Qed.

Lemma cinv_traverse_named (env : traverse_env) (x : variable)
    (defining_expr : named) (a : acc) :
  cinv a -> cinv (traverse_named env x defining_expr a).
Proof.
  intros Hinv. unfold traverse_named.
  destruct defining_expr as [ s | p args | id vs | slot decls vs | w c
                            | m w c | mf mt c ].
  - apply cinv_add_plain; [ apply cinv_note_simple; exact Hinv | plain_tac ].
  - apply cinv_traverse_prim. exact Hinv.
  - apply cinv_record_set; [ | | exact Hinv ].
    + intros base code [Heq | []]. injection Heq as <- _. reflexivity.
    + intros f c [].
  - apply cinv_traverse_closure_of_set. exact Hinv.
  - apply cinv_add_plain; [ apply cinv_note_simple; exact Hinv | plain_tac ].
  - apply cinv_add_plain; [ apply cinv_note_simple; exact Hinv | plain_tac ].
  - apply cinv_add_plain; [ apply cinv_note_simple; exact Hinv | plain_tac ].
Qed.

Lemma cinv_traverse_static_binding (env : traverse_env)
    (binding : static_binding) (a : acc) :
  cinv a -> cinv (traverse_static_binding env binding a).
Proof.
  intros Hinv. unfold traverse_static_binding.
  destruct binding as [ | | s const]; [ exact Hinv | exact Hinv | ].
  destruct const as [ id vs | slot syms decls vs | tag fields
                    | contents | str | [ | f fields] ].
  - apply cinv_record_set; [ | | exact Hinv ].
    + intros base code [Heq | []]. injection Heq as <- _. reflexivity.
    + intros f c [].
  - apply cinv_record_set; [ | | exact Hinv ].
    + intros base code [Heq | []]. injection Heq as <- _. reflexivity.
    + intros f c Hin. apply in_map_iff in Hin.
      destruct Hin as [[f' sym] [Heq _]]. injection Heq as _ <-.
      reflexivity.
  - apply cinv_block_deps. exact Hinv.
  - unfold boxed_number_facts. apply cinv_add_plain; [ exact Hinv | ].
    plain_tac.
  - apply cinv_add_plain; [ exact Hinv | plain_tac ].
  - apply cinv_add_plain; [ exact Hinv | plain_tac ].
  - apply cinv_block_deps. exact Hinv.
Qed.

(* ------------------------------------------------------------------------ *)
(* Code                                                                     *)
(* ------------------------------------------------------------------------ *)

Ltac cid_close :=
  repeat match goal with
  | Hy : (_, ?r) ∈ indexed ?l, Hl : ∀ x, x ∈ ?l -> is_cid x = false
    |- context [is_cid ?r] =>
    rewrite (Hl r (FlambdaReaperLeastProofs.AutoWidgets.indexed_in _ _ _ Hy))
  | H : is_cid ?n = false |- context [is_cid ?n] => rewrite H
  end;
  cbn [is_cid negb andb orb]; try reflexivity.

Lemma create_known_spec (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  (∀ x, x ∈ returns -> is_cid x = false) -> is_cid exn = false ->
  (cinv a ->
   cinv (snd (create_known_arity_call_witness id params returns exn a)))
  ∧ is_cid (fst (create_known_arity_call_witness id params returns exn a))
    = false
  ∧ acc_code_deps (snd (create_known_arity_call_witness id params returns
                          exn a))
    = acc_code_deps a.
Proof.
  intros Hr He. unfold create_known_arity_call_witness. cbn [fresh fst snd].
  split; [ | split; reflexivity ].
  intros Hinv. apply cinv_add_plain; [ apply (cinv_fresh a Hinv) | ].
  plain_tac; cid_close.
Qed.

Lemma witness_deps_spec (id : code_id) (returns : list node) (exn : node)
    (pw : list (list node * node)) (a : acc) :
  (∀ x, x ∈ returns -> is_cid x = false) -> is_cid exn = false ->
  (∀ first w, (first, w) ∈ pw -> is_cid w = false) ->
  (cinv a -> cinv (unknown_arity_witness_deps id returns exn pw a))
  ∧ acc_code_deps (unknown_arity_witness_deps id returns exn pw a)
    = acc_code_deps a.
Proof.
  intros Hr He. revert a. induction pw as [ | [first w] rest IH]; intros a Hw;
    cbn [unknown_arity_witness_deps]; [ auto | ].
  destruct rest as [ | [g nw] rest'].
  - split; [ | reflexivity ]. intros Hinv.
    apply cinv_add_plain; [ apply cinv_add_plain; [ exact Hinv | ] | ];
      plain_tac; cid_close.
  - cbn [fresh]. cbv beta iota zeta.
    pose proof (Hw _ _ (or_intror (or_introl eq_refl))) as Hnw.
    match goal with
    | |- context [unknown_arity_witness_deps id returns exn ?l ?b] =>
      destruct (IH b) as [H1 H2];
      [ intros first' w' Hin; exact (Hw _ _ (or_intror Hin)) | ]
    end.
    split; [ | rewrite H2; reflexivity ].
    intros Hinv. apply H1.
    apply cinv_add_plain; [ | plain_tac; cid_close ].
    apply (cinv_fresh _). apply cinv_add_plain; [ exact Hinv | plain_tac ].
Qed.

Lemma create_unknown_spec (id : code_id) (is_tupled : bool) (ar : arity)
    (params returns : list node) (exn : node) (a : acc) :
  (∀ x, x ∈ returns -> is_cid x = false) -> is_cid exn = false ->
  (cinv a ->
   cinv (snd (create_unknown_arity_call_witnesses id is_tupled ar params
                returns exn a)))
  ∧ (∀ x, x ∈ fst (create_unknown_arity_call_witnesses id is_tupled ar
                     params returns exn a) -> is_cid x = false)
  ∧ acc_code_deps (snd (create_unknown_arity_call_witnesses id is_tupled ar
                          params returns exn a))
    = acc_code_deps a.
Proof.
  intros Hr He. unfold create_unknown_arity_call_witnesses.
  destruct is_tupled.
  - unfold create_unknown_arity_tupled_call_witnesses. cbn [fresh fst snd].
    split; [ | split; [ intros x [<- | []]; reflexivity | reflexivity ] ].
    intros Hinv. apply cinv_add_plain; [ | plain_tac; cid_close ].
    apply (cinv_fresh _), (cinv_fresh _), (cinv_fresh _), Hinv.
  - unfold create_unknown_arity_non_tupled_call_witnesses. cbv zeta.
    set (groups := at_least_one_group _).
    pose proof (fresh_list_made (length groups) a) as Hm.
    pose proof (cinv_fresh_list (length groups) a) as Hf.
    destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec
                (length groups) a) as (_ & Hcd & _).
    destruct (fresh_list (length groups) a) as [ws b] eqn:E.
    cbn [fst snd] in *.
    destruct (witness_deps_spec id returns exn (combine groups ws) b Hr He)
      as [H1 H2].
    { intros first w Hin. exact (Hm _ (in_combine_r _ _ _ _ Hin)). }
    split; [ auto | split; [ exact Hm | rewrite H2; exact Hcd ] ].
Qed.

Lemma cinv_prepare_code (id : code_id) (c : code) (a : acc) :
  cinv a -> cinv (prepare_code id c a).
Proof.
  intros Hinv. unfold prepare_code.
  pose proof (fresh_list_made (length (code_result_kinds c)) a) as Hr.
  pose proof (cinv_fresh_list (length (code_result_kinds c)) a Hinv) as H1.
  destruct (fresh_list (length (code_result_kinds c)) a) as [returns b1].
  cbn [fst snd fresh] in *. cbv beta iota.
  set (b2 := {| acc_next := _ |}).
  assert (H2 : cinv b2) by exact (cinv_fresh _ (cinv_fresh _ H1)).
  pose proof (fresh_list_made (length (code_params_kinds c)) b2) as Hp.
  pose proof (cinv_fresh_list (length (code_params_kinds c)) b2 H2) as H3.
  destruct (fresh_list (length (code_params_kinds c)) b2) as [params b3].
  cbn [fst snd] in *.
  destruct (create_known_spec id params returns
              (Node_made (acc_next b1)) b3 Hr eq_refl) as (H4 & Hk & Hcd4).
  destruct (create_known_arity_call_witness id params returns
              (Node_made (acc_next b1)) b3) as [known b4].
  cbn [fst snd] in *.
  destruct (create_unknown_spec id (code_is_tupled c) (code_params_arity c)
              params returns (Node_made (acc_next b1)) b4 Hr eq_refl)
    as (H5 & Hu & Hcd5).
  destruct (create_unknown_arity_call_witnesses id (code_is_tupled c)
              (code_params_arity c) params returns (Node_made (acc_next b1))
              b4) as [unknown b5].
  cbn [fst snd] in *.
  pose proof (cinv_add_plain [Fact_any_source (Node_code_id id)] b5
                (H5 (H4 H3)) ltac:(plain_tac)) as [C1 C2 C3 C4 C5 C6].
  constructor; cbn [add_code_dep acc_facts acc_code_deps acc_apply_deps
                    acc_closure_deps]; auto.
  intros id' cd n [Heq | Hin]; [ | exact (C4 _ _ _ Hin) ].
  injection Heq as <- <-. cbn [dep_nodes cd_my_closure
    cd_known_arity_call_witness cd_params cd_unknown_arity_call_witnesses].
  intros [<- | [<- | Hn]]; [ reflexivity | exact Hk | ].
  apply in_app_or in Hn as [Hn | Hn]; [ exact (Hp _ Hn) | exact (Hu _ Hn) ].
Qed.

Lemma cinv_prepare_binding (binding : static_binding) (a : acc) :
  cinv a -> cinv (prepare_binding binding a).
Proof.
  destruct binding; cbn [prepare_binding]; auto.
  apply cinv_prepare_code.
Qed.

(* ------------------------------------------------------------------------ *)
(* Calls                                                                    *)
(* ------------------------------------------------------------------------ *)

Ltac in_split H :=
  repeat match type of H with
  | In _ (_ ++ _) => apply in_app_or in H; destruct H as [H | H]
  | In _ (map _ _) =>
    let y := fresh "y" in let Hy := fresh "Hy" in
    apply in_map_iff in H; destruct H as [y [H Hy]]
  | In _ (_ :: _) => destruct H as [H | H]
  | In _ [] => destruct H
  end;
  repeat match goal with y : prod _ _ |- _ => destruct y end.

(* Some fact of [L] is the read of the code id of [b]. *)
Ltac read_in :=
  first
    [ solve [ left; reflexivity ]
    | solve [ apply in_or_app; left; read_in ]
    | solve [ apply in_or_app; right; read_in ]
    | solve [ right; read_in ] ].

Lemma make_known_spec (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  cinv a ->
  cinv (snd (make_known_arity_apply_widget env ap returns exn a))
  ∧ is_cid (fst (make_known_arity_apply_widget env ap returns exn a)) = false
  ∧ acc_code_deps (snd (make_known_arity_apply_widget env ap returns exn a))
    = acc_code_deps a
  ∧ acc_apply_deps (snd (make_known_arity_apply_widget env ap returns exn a))
    = acc_apply_deps a
  ∧ acc_closure_deps
      (snd (make_known_arity_apply_widget env ap returns exn a))
    = acc_closure_deps a.
Proof.
  intros Hinv. unfold make_known_arity_apply_widget.
  cbn [fresh]. cbv beta iota zeta. cbn [fst snd acc_next].
  set (b := {| acc_next := _ |}).
  assert (Hb : cinv b)
    by exact (cinv_fresh _ (cinv_fresh _ (cinv_fresh _ Hinv))).
  destruct (FlambdaReaperLeastProofs.AutoWidgets.note_simples_fields
              (ap_args ap) b) as (_ & N2 & N3 & N4 & N5).
  pose proof (cinv_note_simples (ap_args ap) b Hb) as Hb'.
  set (b' := note_simples _ b) in *. clearbody b'.
  set (c := add_facts _ b').
  assert (Hc : cinv c).
  { apply cinv_add_facts; [ exact Hb' | plain_tac | | ].
    - intros x w H. in_split H; try discriminate H.
      injection H as <- _. read_in.
    - intros y i w H. in_split H; try discriminate H.
      injection H as _ _ <-. eexists. read_in. }
  split; [ apply cinv_cond_alias; [ reflexivity | exact Hc ] | ].
  split; [ reflexivity | ].
  unfold cond_alias. subst c.
  destruct (env_current_code_id env); cbn; rewrite ?N3, ?N4, ?N5; auto.
Qed.

Lemma cinv_unknown_apply_deps (returns : list node) (exn called : node)
    (aw : list (list simple * node)) (a : acc) :
  Fact_any_usage called ∈ acc_facts a ->
  (∀ first w, (first, w) ∈ aw -> is_cid w = false) ->
  cinv a -> cinv (unknown_arity_apply_deps returns exn called aw a)
  ∧ Fact_any_usage called
    ∈ acc_facts (unknown_arity_apply_deps returns exn called aw a)
  ∧ acc_code_deps (unknown_arity_apply_deps returns exn called aw a)
    = acc_code_deps a
  ∧ acc_apply_deps (unknown_arity_apply_deps returns exn called aw a)
    = acc_apply_deps a
  ∧ acc_closure_deps (unknown_arity_apply_deps returns exn called aw a)
    = acc_closure_deps a.
Proof.
  revert a. induction aw as [ | [first w] rest IH]; intros a Hu Hw Hinv;
    cbn [unknown_arity_apply_deps]; [ auto | ].
  pose proof (Hw _ _ (or_introl eq_refl)) as Hw0.
  set (b := add_facts _ a).
  assert (Hb : cinv b).
  { apply cinv_add_facts; [ exact Hinv | plain_tac; cid_close | | ].
    - intros x w' H. in_split H; try discriminate H.
      injection H as <- _. apply in_or_app. right. exact Hu.
    - intros y i w' H. in_split H; try discriminate H.
      injection H as _ _ <-. eexists. read_in. }
  assert (Hub : Fact_any_usage called ∈ acc_facts b)
    by (apply in_or_app; right; exact Hu).
  destruct rest as [ | [g nw] rest'].
  - split; [ apply cinv_add_plain; [ exact Hb | plain_tac; cid_close ] | ].
    split; [ apply in_or_app; right; exact Hub | auto ].
  - cbn [fresh]. cbv beta iota zeta.
    pose proof (Hw _ _ (or_intror (or_introl eq_refl))) as Hnw.
    match goal with
    | |- context [unknown_arity_apply_deps returns exn called ?l ?c] =>
      destruct (IH c) as (H1 & H2 & H3 & H4 & H5)
    end.
    + apply in_or_app. right. exact Hub.
    + intros first' w' Hin. exact (Hw _ _ (or_intror Hin)).
    + apply cinv_add_plain; [ apply (cinv_fresh _ Hb) | plain_tac ].
    + rewrite H3, H4, H5. auto.
Qed.

Lemma make_unknown_spec (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  cinv a ->
  cinv (snd (make_unknown_arity_apply_widget env ap returns exn a))
  ∧ acc_code_deps (snd (make_unknown_arity_apply_widget env ap returns exn
                          a))
    = acc_code_deps a
  ∧ acc_apply_deps (snd (make_unknown_arity_apply_widget env ap returns exn
                           a))
    = acc_apply_deps a
  ∧ acc_closure_deps
      (snd (make_unknown_arity_apply_widget env ap returns exn a))
    = acc_closure_deps a.
Proof.
  intros Hinv. unfold make_unknown_arity_apply_widget.
  cbn [fresh]. cbv beta iota zeta.
  set (b1 := add_fact _ _).
  assert (Hb1 : cinv b1)
    by (apply cinv_add_plain; [ exact (cinv_fresh _ Hinv) | plain_tac ]).
  assert (Hu1 : Fact_any_usage (Node_made (acc_next a)) ∈ acc_facts b1)
    by (left; reflexivity).
  set (groups := at_least_one_group _).
  pose proof (fresh_list_made (length groups) b1) as Hm.
  pose proof (cinv_fresh_list (length groups) b1 Hb1) as Hb2.
  destruct (FlambdaReaperLeastProofs.AutoWidgets.fresh_list_spec
              (length groups) b1) as (F1 & F2 & F3 & F4 & _).
  destruct (fresh_list (length groups) b1) as [ws b2] eqn:E2.
  cbn [fst snd] in *. cbv beta iota zeta.
  destruct (FlambdaReaperLeastProofs.AutoWidgets.note_simples_fields
              (ap_args ap) b2) as (_ & N2 & N3 & N4 & N5).
  pose proof (cinv_note_simples (ap_args ap) b2 Hb2) as Hb3.
  set (b3 := note_simples (ap_args ap) b2) in *. clearbody b3.
  destruct (cinv_unknown_apply_deps returns exn (Node_made (acc_next a))
              (combine groups ws) b3) as (H1 & _ & H3 & H4 & H5).
  { rewrite N2, F1. exact Hu1. }
  { intros first w Hin. exact (Hm _ (in_combine_r _ _ _ _ Hin)). }
  { exact Hb3. }
  set (b4 := unknown_arity_apply_deps _ _ _ _ b3) in *. clearbody b4.
  cbn [fresh]. cbv beta iota.
  assert (Hdeps : acc_code_deps b4 = acc_code_deps a
                  ∧ acc_apply_deps b4 = acc_apply_deps a
                  ∧ acc_closure_deps b4 = acc_closure_deps a).
  { rewrite H3, H4, H5, N3, N4, N5, F2, F3, F4. auto. }
  destruct Hdeps as (D1 & D2 & D3).
  destruct ws as [ | first rest]; cbn [fst snd].
  - split; [ exact (cinv_fresh _ H1) | auto ].
  - split; [ apply cinv_cond_alias;
             [ reflexivity | exact (cinv_fresh _ H1) ]
           | ].
    unfold cond_alias. destruct (env_current_code_id env); cbn; auto.
Qed.

Lemma cinv_push_apply_dep (d : apply_dep) (a : acc) :
  is_cid (ad_apply_closure d) = false -> cinv a ->
  cinv (push_apply_dep d a).
Proof.
  intros Hd [H1 H2 H3 H4 H5 H6]. constructor; cbn; auto.
  intros d' [<- | Hin]; [ exact Hd | auto ].
Qed.

Lemma cinv_add_apply (env : traverse_env) (id : code_id)
    (callee call_widget : node) (only : bool) (a : acc) :
  is_cid callee = false -> cinv a ->
  cinv (add_apply env id callee call_widget only a).
Proof.
  intros Hc Hinv. unfold add_apply. destruct only.
  - cbn [fresh]. cbv beta iota zeta.
    apply cinv_push_apply_dep; [ reflexivity | ].
    apply cinv_add_plain; [ exact (cinv_fresh _ (cinv_fresh _ Hinv)) | ].
    plain_tac.
  - apply cinv_push_apply_dep; [ exact Hc | exact Hinv ].
Qed.

Lemma cinv_traverse_call_kind (env : traverse_env) (ap : apply_expr)
    (exn_arg : node) (return_args : list node) (a : acc) :
  cinv a -> cinv (traverse_call_kind env ap exn_arg return_args a).
Proof.
  intros Hinv. unfold traverse_call_kind. cbv zeta.
  pose proof (cinv_note_simple (ap_callee ap) a Hinv) as Hb.
  set (b := note_simple (ap_callee ap) a) in *. clearbody b.
  destruct (ap_call_kind ap) as [id | | code_ids].
  - destruct (make_known_spec env ap return_args exn_arg b Hb)
      as (H1 & _ & _ & _ & _).
    destruct (make_known_arity_apply_widget env ap return_args exn_arg b)
      as [w c]. cbn [fst snd] in *. cbv beta iota zeta.
    assert (Hc : cinv (add_fact (Fact_accessor w
                                   (Field_call_witness
                                      Known_arity_code_pointer)
                                   (node_of_simple (ap_callee ap))) c))
      by (apply cinv_add_plain; [ exact H1 | plain_tac ]).
    destruct (env_should_preserve_direct_calls env);
      try (apply cinv_add_apply; [ apply is_cid_simple | exact Hc ]).
    destruct (negb _); [ | exact Hc ].
    apply cinv_add_apply; [ apply is_cid_simple | exact Hc ].
  - destruct (make_unknown_spec env ap return_args exn_arg b Hb)
      as (H1 & _ & _ & _).
    destruct (make_unknown_arity_apply_widget env ap return_args exn_arg b)
      as [w c]. cbn [fst snd] in *. cbv beta iota zeta.
    apply cinv_add_plain; [ exact H1 | plain_tac ].
  - destruct (make_known_spec env ap return_args exn_arg b Hb)
      as (H1 & _ & _ & _ & _).
    destruct (make_known_arity_apply_widget env ap return_args exn_arg b)
      as [w c]. cbn [fst snd] in *. cbv beta iota zeta.
    apply cinv_add_plain; [ exact H1 | plain_tac ].
Qed.

Lemma cinv_traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  cinv a -> cinv (traverse_apply env ap a).
Proof.
  intros Hinv. unfold traverse_apply.
  pose proof (cinv_find_cont env (ap_continuation ap)
                (length (unarize (ap_return_arity ap))) a Hinv) as H1.
  destruct (find_cont env (ap_continuation ap) _ a) as [ra b1].
  cbn [snd] in H1. cbv beta iota.
  pose proof (cinv_find_cont env (ap_exn_continuation ap)
                (S (length (ap_exn_extra_args ap))) b1 H1) as H2.
  destruct (find_cont env (ap_exn_continuation ap) _ b1) as [ep b2].
  cbn [snd] in H2. cbv beta iota.
  destruct ep as [ | p extra].
  - cbn [fresh]. cbv beta iota. apply cinv_traverse_call_kind.
    exact (cinv_fresh _ H2).
  - cbv beta iota zeta. apply cinv_traverse_call_kind.
    apply cinv_add_plain; [ apply cinv_note_simples; exact H2 | ].
    plain_tac.
Qed.

Lemma cinv_apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc) :
  cinv a -> cinv (apply_cont_deps env ac a).
Proof.
  intros Hinv. unfold apply_cont_deps. cbv zeta.
  pose proof (cinv_find_cont env (ac_continuation ac) (length (ac_args ac)) a
                Hinv) as H1.
  destruct (find_cont env (ac_continuation ac) _ a) as [params b].
  cbn [snd] in H1. cbv beta iota.
  apply cinv_add_plain; [ apply cinv_note_simples; exact H1 | plain_tac ].
Qed.

Lemma cinv_traverse_switch (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  cinv a -> cinv (traverse_switch env scrutinee arms a).
Proof.
  intros Hinv. unfold traverse_switch. cbv zeta.
  apply cinv_fold; [ | apply cinv_cond_any_usage; exact Hinv ].
  intros [z ac] b _ Hb. apply cinv_apply_cont_deps. exact Hb.
Qed.

Lemma cinv_exn_handler_deps (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  cinv a -> cinv (exn_handler_deps env h a).
Proof.
  intros Hinv. unfold exn_handler_deps.
  destruct h as [[ | p ps] md body]; [ exact Hinv | ].
  destruct (hm_is_exn_handler md); [ | exact Hinv ].
  apply cinv_cond_any_usage. apply cinv_add_plain; [ exact Hinv | plain_tac ].
Qed.

(* ------------------------------------------------------------------------ *)
(* The traversal                                                            *)
(* ------------------------------------------------------------------------ *)

Lemma cinv_traverse :
  (∀ e env a, cinv a -> cinv (traverse env e a))
  ∧ (∀ h env a, cinv a -> cinv (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH env a Hinv. cbn [traverse].
    apply IH. apply cinv_traverse_named. exact Hinv.
  - intros bindings body IHc IH env a Hinv. cbn [traverse]. cbv zeta.
    apply IH. apply cinv_fold; [ | apply cinv_fold; [ | apply cinv_fold ] ].
    + intros [id c | id | s const] b Hin Hb; cbn [traverse_binding];
        [ | exact Hb | exact Hb ].
      destruct c as [params mc ret exn md cbody]. cbn [traverse_code].
      destruct (lookup id (acc_code_deps b)) as [cd | ] eqn:Ecd;
        [ | exact Hb ].
      pose proof (FlambdaMapProofs.lookup_in _ _ _ Ecd) as Hcd.
      apply (IHc id (Code params mc ret exn md cbody) Hin).
      apply cinv_add_plain; [ exact Hb | ].
      plain_tac.
      * rewrite (ci_code_deps b Hb id cd _ Hcd (or_introl eq_refl)).
        reflexivity.
      * apply in_combine_r in Hy.
        rewrite (ci_code_deps b Hb id cd _ Hcd
                   (or_intror (or_intror (in_or_app _ _ _ (or_introl Hy))))).
        reflexivity.
    + intros binding b _ Hb. apply cinv_traverse_static_binding. exact Hb.
    + intros binding b _ Hb. apply cinv_prepare_binding. exact Hb.
    + exact Hinv.
  - intros k h body IHh IH env a Hinv. cbn [traverse]. cbv zeta.
    apply IH. apply cinv_exn_handler_deps. apply IHh. exact Hinv.
  - intros handlers body IHh IH env a Hinv. cbn [traverse]. cbv zeta.
    apply IH. apply cinv_fold; [ | exact Hinv ].
    intros [k h] b Hin Hb. exact (IHh k h Hin _ _ Hb).
  - intros ap env a Hinv. apply cinv_traverse_apply. exact Hinv.
  - intros ac env a Hinv. apply cinv_apply_cont_deps. exact Hinv.
  - intros scrutinee arms env a Hinv. apply cinv_traverse_switch.
    exact Hinv.
  - intros env a Hinv. exact Hinv.
  - intros params md handler IH env a Hinv. cbn [traverse_handler].
    apply IH. exact Hinv.
Qed.

Lemma cinv_resolve (a : acc) (unit_symbols : list symbol) :
  cinv a -> cinv (resolve_delayed_deps unit_symbols a).
Proof.
  intros Hinv. unfold resolve_delayed_deps. cbv zeta.
  assert (Hcd : ∀ id cd n, lookup id (acc_code_deps a) = Some cd ->
                  n ∈ dep_nodes cd -> is_cid n = false).
  { intros id cd n E Hn.
    exact (ci_code_deps a Hinv id cd n (FlambdaMapProofs.lookup_in _ _ _ E)
             Hn). }
  set (b1 := add_facts _ a).
  assert (H1 : cinv b1).
  { apply cinv_add_plain; [ exact Hinv | ].
    intros fct H. apply in_flat_map in H. destruct H as [d [Hd H]].
    pose proof (ci_apply_deps a Hinv d Hd) as Hc.
    unfold apply_dep_facts, alias_for_caller in H.
    destruct (lookup (ad_apply_code_id d) (acc_code_deps a)) as [cd | ]
      eqn:E;
      destruct (ad_function_containing_apply_expr d);
      in_split H; subst; cbv [plain_fact fact_ok negb andb orb];
      rewrite ?Hc, ?(Hcd _ cd _ E (or_intror (or_introl eq_refl)));
      reflexivity. }
  set (b2 := List.fold_left _ (acc_closure_deps b1) b1).
  assert (H2 : cinv b2).
  { apply cinv_fold; [ | exact H1 ].
    intros d b Hd Hb. unfold add_closure_dep.
    pose proof (ci_closure_deps b1 H1 d Hd) as Hc.
    destruct (lookup (cld_closure_code_id d) (acc_code_deps a)) as [cd | ]
      eqn:E.
    - apply cinv_add_plain; [ exact Hb | ].
      pose proof (Hcd _ cd _ E (or_intror (or_introl eq_refl))) as Hk.
      destruct (cd_unknown_arity_call_witnesses cd) as [ | w ws] eqn:Ew.
      + plain_tac.
      + assert (Hw : is_cid w = false).
        { apply (Hcd _ cd w E). right. right. apply in_or_app. right.
          rewrite Ew. left. reflexivity. }
        plain_tac.
    - cbn [fresh]. cbv beta iota.
      apply cinv_add_plain; [ exact (cinv_fresh _ Hb) | plain_tac ]. }
  apply cinv_add_plain; [ exact H2 | plain_tac ].
Qed.

Lemma cinv_unit (flag : reaper_preserve_direct_calls) (u : unit_interface)
    (e : expr) :
  cinv (traverse_unit_acc flag u e).
Proof.
  unfold traverse_unit_acc. cbv zeta. cbn [fresh]. cbv beta iota.
  apply cinv_resolve. apply cinv_add_plain; [ | plain_tac ].
  apply (proj1 cinv_traverse). apply cinv_add_plain; [ | plain_tac ].
  apply (cinv_fresh _), (cinv_fresh _).
  apply cinv_add_plain; [ | plain_tac ].
  constructor; cbn; intros; contradiction.
Qed.

(* ======================================================================== *)
(* The two checks                                                           *)
(* ======================================================================== *)

(* A tuple of the facts of a list is a fact of the list. *)
Ltac fact_of H :=
  let fct := fresh "fct" in let Hf := fresh "Hf" in let Hy := fresh "Hy" in
  destruct (FlambdaReaperLeastProofs.AutoWidgets.facts_of_list_in _ _ _ H)
    as [fct [Hf Hy]];
  destruct fct; cbn in Hy; try contradiction;
  destruct Hy as [Hy | []]; try discriminate Hy;
  try (injection Hy as -> -> ->); try (injection Hy as -> ->);
  try (injection Hy as ->).

(* A fact of a list gives its tuple. *)
Ltac in_facts H :=
  cbn [f_alias f_use f_accessor f_constructor f_argument f_parameter
       f_propagate f_alias_if_any_source f_any_usage f_any_source
       f_any_source_if_used facts_of_list];
  apply in_flat_map; eexists; split; [ exact H | left; reflexivity ].

Theorem traverse_argument_reads : traverse_argument_reads_statement.
Proof.
  intros flag u e. pose proof (cinv_unit flag u e) as Hinv.
  rewrite traverse_unit_acc_facts.
  set (a := traverse_unit_acc flag u e) in *. clearbody a.
  unfold check_argument_reads. apply forallb_forall.
  intros [[y i] b] Hin. cbv beta iota. fact_of Hin.
  destruct (ci_arg a Hinv _ _ _ Hf) as (x & Hx).
  apply existsb_exists. exists (x, cid, b). split.
  - in_facts Hx.
  - rewrite node_eqb_refl. cbn [andb]. apply (proj2 (mem_node_In _ _)).
    pose proof (ci_read a Hinv _ _ Hx) as Hu. in_facts Hu.
Qed.

Section UsedCodeIds.

Variable a : acc.
Hypothesis Hinv : cinv a.
Variable is_local : field -> bool.
Variable T : tables.
Hypothesis Hloc : is_local_slots_only is_local.
Hypothesis Hleast : least_solution is_local (facts_of_list (acc_facts a)) T.

Abbreviation F := (facts_of_list (acc_facts a)).
Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).

Lemma closedC : closed is_local G S.
Proof. exact (FlambdaReaperLeastProofs.least_closed is_local F T Hleast). Qed.

Lemma ok_of (fct : fact) : fct ∈ acc_facts a -> fact_ok fct = true.
Proof. exact (ci_ok a Hinv fct). Qed.

Lemma cid_not_local : is_local cid = false.
Proof.
  destruct (is_local cid) eqn:E; [ | reflexivity ].
  destruct (Hloc _ E) as [(w & Hw) | (fs & Hfs)]; discriminate.
Qed.

(* The constructor of a code id is of the field of the code id, and that
   field is read only into a node with any usage. *)
Lemma cid_constructor (base : node) (f : field) (c : code_id) :
  g_constructor G base f (Node_code_id c) -> f = cid.
Proof.
  intros H. cbn in H. fact_of H. pose proof (ok_of _ Hf) as Hok.
  cbn in Hok. destruct f; try discriminate Hok. reflexivity.
Qed.

Lemma cid_read (x b : node) :
  g_accessor G x cid b -> s_any_usage S x = true.
Proof.
  intros H. cbn in H. fact_of H.
  pose proof (ci_read a Hinv _ _ Hf) as Hu.
  apply (any_usage_input _ _ _ closedC). cbn [graph_of_facts g_any_usage].
  in_facts Hu.
Qed.

(* What is aliased from a code id has any usage. *)
Lemma alias_from_cid (x : node) (c : code_id) :
  s_alias S x (Node_code_id c) = true -> s_any_usage S x = true.
Proof.
  intros H.
  pose proof (FlambdaReaperLeastProofs.least_derivable is_local F T Hleast
                (Atom_alias x (Node_code_id c)) H) as D.
  inversion D; subst;
    repeat match goal with
    | H : g_constructor _ _ _ (Node_code_id _) |- _ =>
      apply cid_constructor in H; subst
    end;
    try match goal with
    | H : is_local cid = true |- _ => rewrite cid_not_local in H; discriminate
    end;
    try (eapply cid_read; eassumption);
    exfalso;
    repeat match goal with
    | H : g_alias _ _ _ |- _ => cbn in H; fact_of H
    | H : g_propagate _ _ _ _ |- _ => cbn in H; fact_of H
    | H : g_alias_if_any_source _ _ _ _ |- _ => cbn in H; fact_of H
    | H : g_argument _ _ _ _ |- _ => cbn in H; fact_of H
    end;
    try match goal with
    | Hf : _ ∈ acc_facts _ |- _ => pose proof (ok_of _ Hf) as Hok;
      cbn in Hok; rewrite ?andb_false_r in Hok; discriminate Hok
    end.
  match goal with
  | H : derivable _ _ _ (Atom_escaping_field _ _) |- _ => inversion H; subst
  end.
  match goal with
  | H : g_constructor _ _ _ (Node_code_id _) |- _ =>
    apply cid_constructor in H; subst
  end.
  rewrite cid_not_local in *. discriminate.
Qed.

Lemma used_code_id (c : code_id) :
  s_has_usage S (Node_code_id c) = true ->
  s_any_usage S (Node_code_id c) = true.
Proof.
  intros H.
  destruct (FlambdaReaperWrapperProofs.has_usage_end is_local F T Hleast _ H)
    as (y & Hp & Hend).
  remember (Node_code_id c) as x eqn:Ex.
  destruct Hp as [x | x y' z Ha _]; subst.
  - destruct Hend as [Hy | [(v & f & Hacc & _) | (from & r & Harg & _)]];
      [ exact Hy | exfalso | exfalso ].
    + cbn in Hacc. fact_of Hacc. pose proof (ok_of _ Hf) as Hok.
      discriminate Hok.
    + cbn in Harg. fact_of Harg. pose proof (ok_of _ Hf) as Hok.
      cbn in Hok. rewrite andb_false_r in Hok. discriminate Hok.
  - exact (any_usage_alias _ _ _ closedC _ _ (alias_from_cid _ _ Ha) Ha).
Qed.

End UsedCodeIds.

Theorem traverse_used_code_ids : traverse_used_code_ids_statement.
Proof.
  intros flag is_local u e T Hloc Hleast. unfold check_used_code_ids.
  apply forallb_forall. intros n Hn. destruct n; try reflexivity.
  exact (used_code_id (traverse_unit_acc flag u e) (cinv_unit flag u e)
           is_local T Hloc Hleast c (proj2 (mem_node_In _ _) Hn)).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem arm_needs_no_source_traversal :
  arm_needs_no_source_traversal_statement.
Proof.
  intros flag is_local u e T cannot_change max_unbox_size lambda_lifting
    dominated to_unbox F code_deps unboxed Hloc Hslots Hfrag Hleast Hcc Hdom
    Hunbox Hwidgets Hknown Hcount Hkinds.
  exact (FlambdaReaperWrapperProofs.arm_needs_no_source flag is_local u e T
           cannot_change max_unbox_size lambda_lifting dominated to_unbox
           Hloc Hslots Hfrag Hleast Hcc Hdom Hunbox Hwidgets
           (traverse_used_code_ids flag is_local u e T Hloc Hleast)
           (traverse_argument_reads flag u e) Hknown Hcount Hkinds).
Qed.

Theorem arm_needs_no_source_auto_traversal :
  arm_needs_no_source_auto_traversal_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting
    dominated to_unbox F code_deps unboxed Hloc Hslots Hfrag Hleast Hcc Hdom
    Hunbox Hknown Hcount Hkinds.
  exact (FlambdaReaperWrapperProofs.arm_needs_no_source_auto is_local u e T
           cannot_change max_unbox_size lambda_lifting dominated to_unbox
           Hloc Hslots Hfrag Hleast Hcc Hdom Hunbox
           (traverse_used_code_ids Auto is_local u e T Hloc Hleast)
           (traverse_argument_reads Auto u e) Hknown Hcount Hkinds).
Qed.

Theorem indirect_arm_needs_no_source_traversal :
  indirect_arm_needs_no_source_traversal_statement.
Proof.
  intros flag is_local u e T cannot_change max_unbox_size lambda_lifting
    dominated to_unbox F code_deps unboxed Hloc Hslots Hfrag Hleast Hcc Hdom
    Hunbox Hwidgets Hknown Huw Hux Hcount Hkinds.
  exact (FlambdaReaperWrapperProofs.indirect_arm_needs_no_source flag
           is_local u e T cannot_change max_unbox_size lambda_lifting
           dominated to_unbox Hloc Hslots Hfrag Hleast Hcc Hdom Hunbox
           Hwidgets (traverse_used_code_ids flag is_local u e T Hloc Hleast)
           (traverse_argument_reads flag u e) Hknown Huw Hux Hcount Hkinds).
Qed.

Theorem kept_result_never_unboxed_traversal :
  kept_result_never_unboxed_traversal_statement.
Proof.
  intros flag is_local u e T cannot_change max_unbox_size lambda_lifting
    dominated to_unbox F code_deps unboxed Hloc Hslots Hfrag Hleast Hcc Hdom
    Hunbox Hwidgets Hknown Hcount Hkinds.
  exact (FlambdaReaperWrapperProofs.kept_result_never_unboxed flag is_local
           u e T cannot_change max_unbox_size lambda_lifting dominated
           to_unbox Hloc Hslots Hfrag Hleast Hcc Hdom Hunbox Hwidgets
           (traverse_used_code_ids flag is_local u e T Hloc Hleast)
           (traverse_argument_reads flag u e) Hknown Hcount Hkinds).
Qed.
