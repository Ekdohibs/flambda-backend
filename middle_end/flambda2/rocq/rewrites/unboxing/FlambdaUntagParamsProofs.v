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


(* Proofs of the statement of FlambdaUntagParamsSpec.v. Nothing here
   needs review.

   The term and its result run in step under a relation of states
   ([state_rel]): the same expression, where neither run is in a term
   that the pass rewrote; or an expression that the test accepts and
   the same rewritten, under variables that agree but for the untagged
   ones, which hold a tagged immediate in the first run and the naked
   immediate of the same number in the second, and under continuations
   that are closures of a handler and of that handler rewritten. A
   variable of the tags holds, in the first run, the tagged immediate
   of the number that the variable it tags holds naked ([tags_ok]);
   that one is not untagged, so it holds the same in the second. That
   the result refines the term is then
   FlambdaSemanticsProofs.simulation_refines. *)

From Stdlib Require Import ZArith List Bool Arith Lia Permutation.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaPrimitiveModel FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaUnusedParamsProofs FlambdaUntagParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

Lemma kept_from_nil {A : Type} (l : list A) :
  ∀ i : nat, kept_from [] i l = l.
Proof.
  induction l as [ | a l IH]; intros i; cbn [kept_from at_position existsb];
    [ reflexivity | rewrite IH; reflexivity ].
Qed.

Lemma moved_from_nil {A : Type} (l : list A) :
  ∀ i : nat, moved_from [] i l = [].
Proof.
  induction l as [ | a l IH]; intros i; cbn [moved_from at_position existsb];
    [ reflexivity | apply IH ].
Qed.

Lemma kept_from_map {A B : Type} (f : A -> B) (us : list nat) (l : list A) :
  ∀ i : nat, kept_from us i (map f l) = map f (kept_from us i l).
Proof.
  induction l as [ | a l IH]; intros i; cbn [kept_from map];
    [ reflexivity | ].
  destruct (at_position us i); cbn [map]; rewrite IH; reflexivity.
Qed.

Lemma moved_from_map {A B : Type} (f : A -> B) (us : list nat) (l : list A) :
  ∀ i : nat, moved_from us i (map f l) = map f (moved_from us i l).
Proof.
  induction l as [ | a l IH]; intros i; cbn [moved_from map];
    [ reflexivity | ].
  destruct (at_position us i); cbn [map]; rewrite IH; reflexivity.
Qed.

Lemma kept_from_length {A B : Type} (us : list nat) (l : list A) :
  ∀ (l' : list B) (i : nat),
    length l = length l' ->
    length (kept_from us i l) = length (kept_from us i l').
Proof.
  induction l as [ | a l IH]; intros [ | b l'] i Hlength;
    try discriminate Hlength; cbn [kept_from]; [ reflexivity | ].
  injection Hlength as Hlength.
  destruct (at_position us i); cbn [length]; rewrite (IH l' (S i) Hlength);
    reflexivity.
Qed.

Lemma moved_from_length {A B : Type} (us : list nat) (l : list A) :
  ∀ (l' : list B) (i : nat),
    length l = length l' ->
    length (moved_from us i l) = length (moved_from us i l').
Proof.
  induction l as [ | a l IH]; intros [ | b l'] i Hlength;
    try discriminate Hlength; cbn [moved_from]; [ reflexivity | ].
  injection Hlength as Hlength.
  destruct (at_position us i); cbn [length]; rewrite (IH l' (S i) Hlength);
    reflexivity.
Qed.

Lemma among_moved_from (us : list nat) (x : variable) (l : list variable) :
  ∀ i : nat, among x (moved_from us i l) = true -> among x l = true.
Proof.
  induction l as [ | a l IH]; intros i; cbn [moved_from];
    [ intros H; exact H | ].
  rewrite among_cons.
  destruct (at_position us i).
  - rewrite among_cons. intros H. apply orb_true_iff in H as [H | H];
      [ rewrite H; reflexivity | rewrite (IH _ H); apply orb_true_r ].
  - intros H. rewrite (IH _ H). apply orb_true_r.
Qed.

(* The elements kept and then those moved are those of the list. *)
Lemma kept_moved_permutation {A : Type} (us : list nat) (l : list A) :
  ∀ i : nat,
    Permutation l (kept_from us i l ++ rev (moved_from us i l)).
Proof.
  induction l as [ | a l IH]; intros i; cbn [kept_from moved_from];
    [ apply Permutation_refl | ].
  destruct (at_position us i).
  - cbn [rev]. rewrite app_assoc.
    exact
      (Permutation_trans (perm_skip a (IH (S i)))
         (Permutation_cons_append _ a)).
  - cbn [app]. apply perm_skip, IH.
Qed.

Lemma distinct_untag_params_list (us : list nat) (params : list variable) :
  distinct params = true -> distinct (untag_params_list us params) = true.
Proof.
  rewrite !distinct_NoDup. intros Hdistinct. unfold untag_params_list.
  exact (Permutation_NoDup (kept_moved_permutation us params 0) Hdistinct).
Qed.

(* With no position, the rewrite of the arguments, of the parameters
   and of the metadata changes nothing. *)
Lemma untag_args_nil (tagged : tags) (args : list simple) :
  untag_args [] tagged args = args.
Proof.
  unfold untag_args. rewrite kept_from_nil, moved_from_nil. apply app_nil_r.
Qed.

Lemma untag_params_list_nil (params : list variable) :
  untag_params_list [] params = params.
Proof.
  unfold untag_params_list. rewrite kept_from_nil, moved_from_nil.
  apply app_nil_r.
Qed.

Lemma untag_metadata_nil (metadata : handler_metadata) :
  untag_metadata [] metadata = metadata.
Proof.
  destruct metadata as [kinds is_exn]. unfold untag_metadata.
  cbn [hm_params_kinds hm_is_exn_handler].
  rewrite kept_from_nil, moved_from_nil. cbn [rev map]. rewrite app_nil_r.
  reflexivity.
Qed.

(* ======================================================================== *)
(* Values and variables                                                     *)
(* ======================================================================== *)

Definition untag_value (v : value) : value :=
  match v with
  | Value_const (Tagged_immediate i) => Value_const (Naked_immediate i)
  | _ => v
  end.

Definition is_tagged (v : value) : Prop :=
  ∃ i : Z, v = Value_const (Tagged_immediate i).

Definition untag_values (us : list nat) (vs : list value) : list value :=
  kept_from us 0 vs ++ rev (map untag_value (moved_from us 0 vs)).

Lemma untag_values_nil (vs : list value) : untag_values [] vs = vs.
Proof.
  unfold untag_values. rewrite kept_from_nil, moved_from_nil. apply app_nil_r.
Qed.

(* The variables of the two runs: an untagged one holds a tagged
   immediate in the first and the naked immediate of the same number in
   the second; the others hold the same. *)
Definition vars_rel (untagged : list variable) (vars vars' : env) : Prop :=
  ∀ x : variable,
    if among x untagged
    then
      ∃ i : Z,
        lookup x vars = Some (Value_const (Tagged_immediate i))
        ∧ lookup x vars' = Some (Value_const (Naked_immediate i))
    else lookup x vars = lookup x vars'.

Lemma vars_rel_nil (vars : env) : vars_rel [] vars vars.
Proof. intros x. reflexivity. Qed.

Lemma vars_rel_cons (untagged : list variable) (vars vars' : env)
    (y : variable) (v : value) :
  vars_rel untagged vars vars' ->
  vars_rel (without [y] untagged) ((y, v) :: vars) ((y, v) :: vars').
Proof.
  intros Hvars x. rewrite among_without, among_cons, !lookup_cons.
  cbn [among existsb]. rewrite orb_false_r.
  destruct (Pos.eqb x y) eqn:Hxy.
  - rewrite andb_false_r. reflexivity.
  - rewrite andb_true_r. exact (Hvars x).
Qed.

(* A variable that the test says is not mentioned is not untagged. *)
Lemma none_in_not_among (untagged : list variable)
    (mentions : variable -> bool) (x : variable) :
  none_in untagged mentions = true ->
  mentions x = true ->
  among x untagged = false.
Proof.
  unfold none_in. intros Hnone Hx.
  destruct (among x untagged) eqn:Hamong; [ | reflexivity ].
  apply among_In in Hamong.
  rewrite forallb_forall in Hnone. specialize (Hnone x Hamong).
  rewrite Hx in Hnone. discriminate Hnone.
Qed.

(* What the tags say holds of the variables of the first run: the
   variable of a tag holds the tagged immediate of the number that the
   other variable holds naked, and the other is not untagged, so it
   holds the same in the second run. *)
Definition tags_ok (untagged : list variable) (tagged : tags) (vars : env)
    : Prop :=
  ∀ v n : variable,
    In (v, n) tagged ->
    among n untagged = false
    ∧ ∃ i : Z,
        lookup v vars = Some (Value_const (Tagged_immediate i))
        ∧ lookup n vars = Some (Value_const (Naked_immediate i)).

Lemma tags_ok_nil (untagged : list variable) (vars : env) :
  tags_ok untagged [] vars.
Proof. intros v n []. Qed.

Lemma in_tags_without (xs : list variable) (tagged : tags) (v n : variable) :
  In (v, n) (tags_without xs tagged) ->
  In (v, n) tagged ∧ among v xs = false ∧ among n xs = false.
Proof.
  unfold tags_without. rewrite filter_In. intros [Hin Hf].
  cbn beta iota in Hf. apply andb_true_iff in Hf as [Hv Hn].
  apply negb_true_iff in Hv. apply negb_true_iff in Hn.
  split; [ exact Hin | split; assumption ].
Qed.

(* What a defining expression that tags a variable is, and what it
   gives. *)
Lemma tag_of_some (d : named) (n : variable) :
  tag_of d = Some n -> d = Prim (Unary Tag_immediate) [Var n].
Proof.
  destruct d as [s | [[] | | | | ] [ | [x | sym | c] [ | a args]] | | | | | ];
    simpl; try discriminate.
  intros [= ->]. reflexivity.
Qed.

Lemma eval_tagging (statics : static_env) (vars : env) (n : variable)
    (v : value) :
  eval_named statics vars (Prim (Unary Tag_immediate) [Var n]) = Some v ->
  ∃ i : Z,
    lookup n vars = Some (Value_const (Naked_immediate i))
    ∧ v = Value_const (Tagged_immediate i).
Proof.
  cbn [eval_named eval_simples eval_simple].
  destruct (lookup n vars) as [w | ]; [ | discriminate ].
  simpl. rewrite eval_prim_unary. cbn [eval_unary_primitive].
  destruct (int_of_value Standard_naked_immediate w) as [i | ] eqn:Hint;
    [ | discriminate ].
  intros [= <-]. exists i.
  apply int_of_value_const in Hint as [-> _]. split; reflexivity.
Qed.

(* The tags under a [Let] hold of the variables with the one bound. *)
Lemma tags_ok_let (untagged : list variable) (tagged : tags)
    (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
    (vars : env) (x : variable) (d : named) (v : value) :
  named_ok untagged d = true ->
  tags_ok untagged tagged vars ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  tags_ok (without [x] untagged) (tags_under_let x d tagged)
    ((x, v) :: vars).
Proof.
  intros Hd Htags Hv.
  assert (Hrest :
    tags_ok (without [x] untagged) (tags_without [x] tagged)
      ((x, v) :: vars)).
  { intros v0 n0 Hin. apply in_tags_without in Hin as (Hin & Hv0 & Hn0).
    destruct (Htags v0 n0 Hin) as (Hnot & i & Hv0' & Hn0').
    rewrite among_cons in Hv0, Hn0. cbn [among existsb] in Hv0, Hn0.
    rewrite orb_false_r in Hv0, Hn0.
    split.
    - rewrite among_without, Hnot. reflexivity.
    - exists i. rewrite !lookup_cons, Hv0, Hn0. split; assumption. }
  unfold tags_under_let.
  destruct (tag_of d) as [n | ] eqn:Htag; [ | exact Hrest ].
  destruct (Pos.eqb_spec n x) as [-> | Hne]; [ exact Hrest | ].
  intros v0 n0 [[= <- <-] | Hin]; [ | exact (Hrest v0 n0 Hin) ].
  apply tag_of_some in Htag. subst d.
  (* the tagging reads no cell *)
  apply exec_named_pure in Hv as (Hv & _ & _); [ | reflexivity ].
  destruct (eval_tagging statics vars n v Hv) as (i & Hn & ->).
  assert (Hnot : among n untagged = false).
  { unfold named_ok, reads_untagged in Hd. cbn [untag_of orb] in Hd.
    apply (none_in_not_among untagged _ n Hd).
    cbn. rewrite Pos.eqb_refl. reflexivity. }
  split.
  - rewrite among_without, Hnot. reflexivity.
  - exists i. rewrite !lookup_cons, Pos.eqb_refl.
    destruct (Pos.eqb_spec n x) as [Heq | _]; [ contradiction | ].
    split; [ reflexivity | exact Hn ].
Qed.

Lemma eval_simple_plain (untagged : list variable) (vars vars' : env)
    (s : simple) :
  vars_rel untagged vars vars' ->
  none_in untagged (fun p => var_in_simple p s) = true ->
  eval_simple vars' s = eval_simple vars s.
Proof.
  intros Hvars Hnone. apply eval_simple_agree. intros x Hx.
  pose proof (Hvars x) as Hx'.
  rewrite
    (none_in_not_among untagged _ x Hnone (var_in_simple_sound x s Hx))
    in Hx'.
  exact Hx'.
Qed.

Lemma eval_simples_plain (untagged : list variable) (vars vars' : env)
    (args : list simple) :
  vars_rel untagged vars vars' ->
  none_in untagged (fun p => var_in_simples p args) = true ->
  eval_simples vars' args = eval_simples vars args.
Proof.
  intros Hvars Hnone. apply eval_simples_agree. intros x Hx.
  pose proof (Hvars x) as Hx'.
  rewrite
    (none_in_not_among untagged _ x Hnone (var_in_simples_sound x args Hx))
    in Hx'.
  exact Hx'.
Qed.

Lemma exec_named_plain (untagged : list variable) (statics : static_env)
    (hp : heap) (rd : read_record) (vars vars' : env) (d : named) :
  vars_rel untagged vars vars' ->
  none_in untagged (fun p => var_in_named p d) = true ->
  exec_named statics hp rd vars' d = exec_named statics hp rd vars d.
Proof.
  intros Hvars Hnone. apply exec_named_agree. intros x Hx.
  pose proof (Hvars x) as Hx'.
  rewrite (none_in_not_among untagged _ x Hnone (var_in_named_sound x d Hx))
    in Hx'.
  exact Hx'.
Qed.

Lemma eval_static_bindings_plain (untagged : list variable)
    (vars vars' : env) (bindings : list static_binding)
    (statics : static_env) :
  vars_rel untagged vars vars' ->
  none_in untagged (fun p => existsb (var_in_static_binding p) bindings)
  = true ->
  eval_static_bindings vars' bindings statics
  = eval_static_bindings vars bindings statics.
Proof.
  intros Hvars Hnone. apply eval_static_bindings_agree. intros x Hx.
  pose proof (Hvars x) as Hx'.
  rewrite (none_in_not_among untagged _ x Hnone) in Hx'; [ exact Hx' | ].
  apply Exists_exists in Hx. destruct Hx as (binding & Hin & Hfree).
  apply existsb_exists. exists binding.
  split; [ exact Hin | apply var_in_static_binding_sound, Hfree ].
Qed.

(* ======================================================================== *)
(* The arguments of an application                                          *)
(* ======================================================================== *)

Lemma eval_simples_split (us : list nat) (vars : env) (args : list simple) :
  ∀ (i : nat) (vs : list value),
    eval_simples vars args = Some vs ->
    eval_simples vars (kept_from us i args) = Some (kept_from us i vs)
    ∧ eval_simples vars (moved_from us i args) = Some (moved_from us i vs).
Proof.
  induction args as [ | s args IH]; intros i vs; simpl.
  - intros [= <-]. split; reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ] eqn:Hrest;
      [ | discriminate ].
    intros [= <-]. destruct (IH (S i) rest eq_refl) as [Hkept Hmoved].
    cbn [kept_from moved_from].
    destruct (at_position us i); simpl; rewrite ?Hs, ?Hkept, ?Hmoved;
      split; reflexivity.
Qed.

Lemma eval_simples_rev (vars : env) (args : list simple) :
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    eval_simples vars (rev args) = Some (rev vs).
Proof.
  induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ] eqn:Hrest;
      [ | discriminate ].
    intros [= <-]. rewrite eval_simples_app, (IH rest eq_refl). simpl.
    rewrite Hs. reflexivity.
Qed.

(* The arguments at the untagged positions: each is a tagged
   immediate in the first run and the naked one in the second. *)
Lemma eval_simples_untagged (untagged : list variable) (tagged : tags)
    (vars vars' : env) (args : list simple) :
  vars_rel untagged vars vars' ->
  tags_ok untagged tagged vars ->
  forallb (untagged_arg untagged tagged) args = true ->
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    eval_simples vars' (map (untag_simple tagged) args)
    = Some (map untag_value vs)
    ∧ Forall is_tagged vs.
Proof.
  intros Hvars Htags.
  induction args as [ | s args IH]; intros Hok vs; simpl.
  - intros [= <-]. split; [ reflexivity | constructor ].
  - simpl in Hok. apply andb_true_iff in Hok as [Hs Hargs].
    destruct (eval_simple vars s) as [v | ] eqn:Hv; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ] eqn:Hrest;
      [ | discriminate ].
    intros [= <-]. destruct (IH Hargs rest eq_refl) as [Heval Htagged].
    rewrite Heval.
    assert (Hhead :
      eval_simple vars' (untag_simple tagged s) = Some (untag_value v)
      ∧ is_tagged v).
    { destruct s as [x | sym | c]; cbn [untagged_arg] in Hs;
        try discriminate Hs.
      - cbn [untag_simple].
        destruct (lookup x tagged) as [n | ] eqn:Hn.
        + destruct (Htags x n (lookup_in x n tagged Hn))
            as (Hnot & i & Hx & Hn').
          simpl in Hv. rewrite Hx in Hv. injection Hv as <-.
          pose proof (Hvars n) as Hrel. rewrite Hnot in Hrel.
          cbn [eval_simple untag_value]. rewrite <- Hrel.
          split; [ exact Hn' | exists i; reflexivity ].
        + rewrite orb_false_r in Hs.
          pose proof (Hvars x) as Hx. rewrite Hs in Hx.
          destruct Hx as (i & Hx & Hx'). simpl in Hv. rewrite Hx in Hv.
          injection Hv as <-. simpl. rewrite Hx'.
          split; [ reflexivity | exists i; reflexivity ].
      - destruct c; try discriminate Hs. simpl in Hv. injection Hv as <-.
        split; [ reflexivity | eexists; reflexivity ]. }
    destruct Hhead as [Hhead Hv_tagged]. rewrite Hhead.
    split; [ reflexivity | constructor; assumption ].
Qed.

(* The arguments of an application that the test accepts. *)
Lemma eval_untag_args (untagged : list variable) (tagged : tags)
    (us : list nat) (vars vars' : env) (args : list simple)
    (vs : list value) :
  vars_rel untagged vars vars' ->
  tags_ok untagged tagged vars ->
  args_ok untagged tagged us args = true ->
  eval_simples vars args = Some vs ->
  eval_simples vars' (untag_args us tagged args) = Some (untag_values us vs)
  ∧ Forall is_tagged (moved_from us 0 vs).
Proof.
  intros Hvars Htags Hok Hvs. unfold args_ok in Hok.
  apply andb_true_iff in Hok as [Hkept Hmoved].
  destruct (eval_simples_split us vars args 0 vs Hvs) as [Hk Hm].
  destruct
    (eval_simples_untagged untagged tagged vars vars' _ Hvars Htags Hmoved _
       Hm)
    as [Hm' Htagged].
  split; [ | exact Htagged ].
  unfold untag_args, untag_values.
  rewrite eval_simples_app,
    (eval_simples_plain untagged vars vars' _ Hvars Hkept), Hk,
    (eval_simples_rev vars' _ _ Hm').
  reflexivity.
Qed.

(* ======================================================================== *)
(* The parameters of a handler                                              *)
(* ======================================================================== *)

Lemma lookup_combine_none (x : variable) (params : list variable) :
  ∀ vs : list value,
    among x params = false -> lookup x (combine params vs) = None.
Proof.
  induction params as [ | p params IH]; intros [ | v vs];
    cbn [combine lookup]; try (intros _; reflexivity).
  rewrite among_cons. intros Hx. apply orb_false_iff in Hx as [Hne Hx].
  rewrite Hne. exact (IH vs Hx).
Qed.

Lemma lookup_combine_some (x : variable) (params : list variable) :
  ∀ vs : list value,
    length params = length vs ->
    among x params = true ->
    ∃ v : value, lookup x (combine params vs) = Some v.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] Hlength;
    try discriminate Hlength; cbn [combine lookup]; [ discriminate | ].
  rewrite among_cons. injection Hlength as Hlength.
  destruct (Pos.eqb x p); [ intros _; exists v; reflexivity | ].
  exact (IH vs Hlength).
Qed.

(* What parameters bound to values hold. *)
Lemma lookup_bind_params (params : list variable) :
  ∀ (vs : list value) (vars bound : env) (x : variable),
    distinct params = true ->
    bind_params params vs vars = Some bound ->
    lookup x bound
    = match lookup x (combine params vs) with
      | Some v => Some v
      | None => lookup x vars
      end.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars bound x;
    cbn [bind_params]; try discriminate.
  - intros _ [= <-]. reflexivity.
  - cbn [distinct]. intros Hdistinct Hbind.
    apply andb_true_iff in Hdistinct as [Hp Hdistinct].
    apply negb_true_iff in Hp.
    rewrite (IH vs _ bound x Hdistinct Hbind). cbn [combine].
    rewrite !lookup_cons.
    destruct (Pos.eqb_spec x p) as [-> | Hne]; [ | reflexivity ].
    rewrite (lookup_combine_none p params vs Hp). reflexivity.
Qed.

Lemma bind_params_of_length (params : list variable) :
  ∀ (vs : list value) (vars : env),
    length params = length vs ->
    ∃ bound : env, bind_params params vs vars = Some bound.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars Hlength;
    try discriminate Hlength; cbn [bind_params].
  - exists vars. reflexivity.
  - injection Hlength as Hlength. exact (IH vs _ Hlength).
Qed.

Lemma bind_params_same_length (params : list variable) :
  ∀ (vs : list value) (vars bound : env),
    bind_params params vs vars = Some bound -> length params = length vs.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars bound;
    cbn [bind_params]; try discriminate; [ reflexivity | ].
  intros Hbind. cbn [length]. rewrite (IH vs _ bound Hbind). reflexivity.
Qed.

Lemma combine_snoc {A B : Type} (l : list A) :
  ∀ (l' : list B) (a : A) (b : B),
    length l = length l' ->
    combine (l ++ [a]) (l' ++ [b]) = combine l l' ++ [(a, b)].
Proof.
  induction l as [ | x l IH]; intros [ | y l'] a b Hlength;
    try discriminate Hlength; simpl; [ reflexivity | ].
  injection Hlength as Hlength. rewrite (IH l' a b Hlength). reflexivity.
Qed.

(* The parameters and the values in the order of the rewrite: each
   parameter has the value that it had, untagged if it is one of those
   moved. *)
Lemma lookup_combine_moved (us : list nat) (g : value -> value)
    (x : variable) (params : list variable) :
  ∀ (vs : list value) (i : nat),
    distinct params = true ->
    length params = length vs ->
    lookup x
      (combine (kept_from us i params ++ rev (moved_from us i params))
         (kept_from us i vs ++ rev (map g (moved_from us i vs))))
    = match lookup x (combine params vs) with
      | Some v =>
        Some (if among x (moved_from us i params) then g v else v)
      | None => None
      end.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] i Hdistinct Hlength;
    try discriminate Hlength; [ reflexivity | ].
  cbn [distinct] in Hdistinct.
  apply andb_true_iff in Hdistinct as [Hp Hdistinct].
  apply negb_true_iff in Hp. injection Hlength as Hlength.
  specialize (IH vs (S i) Hdistinct Hlength).
  cbn [kept_from moved_from combine]. rewrite lookup_cons.
  destruct (at_position us i).
  - cbn [rev map]. rewrite !app_assoc, combine_snoc, lookup_app, IH.
    2: { rewrite !length_app, !length_rev, length_map.
         rewrite (kept_from_length us params vs (S i) Hlength),
           (moved_from_length us params vs (S i) Hlength).
         reflexivity. }
    rewrite among_cons. cbn [lookup].
    destruct (Pos.eqb_spec x p) as [-> | Hne].
    + rewrite (lookup_combine_none p params vs Hp). reflexivity.
    + destruct (lookup x (combine params vs)); reflexivity.
  - cbn [app combine]. rewrite lookup_cons.
    destruct (Pos.eqb_spec x p) as [-> | Hne]; [ | exact IH ].
    destruct (among p (moved_from us (S i) params)) eqn:Hmoved;
      [ | reflexivity ].
    apply among_moved_from in Hmoved. congruence.
Qed.

(* The value of a parameter that is moved is one of the values
   moved. *)
Lemma lookup_combine_moved_value (us : list nat) (Q : value -> Prop)
    (x : variable) (params : list variable) :
  ∀ (vs : list value) (i : nat) (v : value),
    distinct params = true ->
    Forall Q (moved_from us i vs) ->
    lookup x (combine params vs) = Some v ->
    among x (moved_from us i params) = true ->
    Q v.
Proof.
  induction params as [ | p params IH]; intros [ | w vs] i v Hdistinct;
    try discriminate.
  cbn [distinct] in Hdistinct.
  apply andb_true_iff in Hdistinct as [Hp Hdistinct].
  apply negb_true_iff in Hp.
  cbn [moved_from combine]. rewrite lookup_cons.
  destruct (at_position us i).
  - intros HQ. inversion HQ as [ | w0 rest Hw Hrest]; subst.
    rewrite among_cons.
    destruct (Pos.eqb x p); [ intros [= <-] _; exact Hw | ].
    exact (IH vs (S i) v Hdistinct Hrest).
  - intros HQ.
    destruct (Pos.eqb_spec x p) as [-> | Hne];
      [ | exact (IH vs (S i) v Hdistinct HQ) ].
    intros _ Hmoved. apply among_moved_from in Hmoved. congruence.
Qed.

(* The handler entered: the variables of the two runs are related
   again, with the parameters moved untagged. *)
Lemma bind_params_rel (untagged : list variable) (us : list nat)
    (params : list variable) (vs : list value) (vars vars' bound : env) :
  distinct params = true ->
  vars_rel untagged vars vars' ->
  Forall is_tagged (moved_from us 0 vs) ->
  bind_params params vs vars = Some bound ->
  ∃ bound' : env,
    bind_params (untag_params_list us params) (untag_values us vs) vars'
    = Some bound'
    ∧ vars_rel (untagged_under us params untagged) bound bound'.
Proof.
  intros Hdistinct Hvars Htagged Hbind.
  pose proof (bind_params_same_length params vs vars bound Hbind) as Hlength.
  pose proof (distinct_untag_params_list us params Hdistinct) as Hdistinct'.
  destruct
    (bind_params_of_length (untag_params_list us params)
       (untag_values us vs) vars')
    as [bound' Hbind'].
  { unfold untag_params_list, untag_values.
    rewrite !length_app, !length_rev, length_map.
    rewrite (kept_from_length us params vs 0 Hlength),
      (moved_from_length us params vs 0 Hlength).
    reflexivity. }
  exists bound'. split; [ exact Hbind' | ].
  intros x.
  rewrite (lookup_bind_params params vs vars bound x Hdistinct Hbind),
    (lookup_bind_params _ _ vars' bound' x Hdistinct' Hbind').
  unfold untag_params_list, untag_values.
  rewrite (lookup_combine_moved us untag_value x params vs 0 Hdistinct Hlength).
  unfold untagged_under. rewrite among_app, among_without.
  destruct (among x (moved_from us 0 params)) eqn:Hmoved; cbn [orb].
  - destruct
      (lookup_combine_some x params vs Hlength
         (among_moved_from us x params 0 Hmoved))
      as [v Hv].
    rewrite Hv.
    destruct
      (lookup_combine_moved_value us is_tagged x params vs 0 v Hdistinct
         Htagged Hv Hmoved)
      as [i ->].
    exists i. split; reflexivity.
  - destruct (among x params) eqn:Hparam.
    + rewrite andb_false_r.
      destruct (lookup_combine_some x params vs Hlength Hparam) as [v Hv].
      rewrite Hv. reflexivity.
    + rewrite andb_true_r, (lookup_combine_none x params vs Hparam).
      exact (Hvars x).
Qed.

(* The tags that are left under the parameters hold of the variables
   of the handler entered. *)
Lemma tags_ok_params (untagged : list variable) (tagged : tags)
    (us : list nat) (params : list variable) (vs : list value)
    (vars bound : env) :
  distinct params = true ->
  tags_ok untagged tagged vars ->
  bind_params params vs vars = Some bound ->
  tags_ok (untagged_under us params untagged) (tags_without params tagged)
    bound.
Proof.
  intros Hdistinct Htags Hbind v n Hin.
  apply in_tags_without in Hin as (Hin & Hv & Hn).
  destruct (Htags v n Hin) as (Hnot & i & Hv' & Hn').
  split.
  - unfold untagged_under. rewrite among_app, among_without, Hnot.
    cbn [andb]. rewrite orb_false_r.
    destruct (among n (moved_from us 0 params)) eqn:Hmoved;
      [ | reflexivity ].
    rewrite (among_moved_from us n params 0 Hmoved) in Hn. discriminate Hn.
  - exists i.
    rewrite !(lookup_bind_params params vs vars bound _ Hdistinct Hbind),
      (lookup_combine_none v params vs Hv),
      (lookup_combine_none n params vs Hn).
    split; assumption.
Qed.

(* The kinds of the values in the order of the rewrite are those of the
   metadata rewritten. *)
Lemma kinds_untagged (us : list nat) (vs : list value)
    (metadata : handler_metadata) :
  Forall is_tagged (moved_from us 0 vs) ->
  kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata) = true ->
  kinds_eqb (map kind_of_value (untag_values us vs))
    (hm_params_kinds (untag_metadata us metadata))
  = true.
Proof.
  intros Htagged Hkinds. apply kinds_eqb_eq in Hkinds.
  unfold untag_values, untag_metadata. cbn [hm_params_kinds].
  rewrite <- Hkinds, map_app, kept_from_map, moved_from_map, !map_rev.
  assert (Hmoved :
    map kind_of_value (map untag_value (moved_from us 0 vs))
    = map (fun _ => untagged_kind)
        (map kind_of_value (moved_from us 0 vs))).
  { induction Htagged as [ | v rest [i ->] _ IH]; [ reflexivity | ].
    cbn [map]. rewrite IH. reflexivity. }
  rewrite Hmoved. apply kinds_eqb_refl.
Qed.

(* ======================================================================== *)
(* What the rewrite keeps of the syntax                                     *)
(* ======================================================================== *)

Fixpoint untag_expr_binds_statics (sites active : positions)
    (untagged : list variable) (tagged : tags) (e : expr) {struct e} :
  binds_statics (untag_expr sites active untagged tagged e) = binds_statics e
with untag_handler_binds_statics (sites active : positions)
    (untagged : list variable) (tagged : tags) (us : list nat)
    (h : continuation_handler) {struct h} :
  handler_binds_statics (untag_handler sites active untagged tagged us h)
  = handler_binds_statics h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [untag_expr binds_statics]; try reflexivity.
    + apply untag_expr_binds_statics.
    + rewrite untag_expr_binds_statics. reflexivity.
    + rewrite untag_handler_binds_statics, untag_expr_binds_statics.
      reflexivity.
    + rewrite untag_expr_binds_statics. f_equal.
      generalize (active_under sites handlers active). intros active'.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map existsb]. rewrite untag_handler_binds_statics, IH.
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [untag_handler handler_binds_statics].
    apply untag_expr_binds_statics.
Qed.

Definition untag_handlers (sites active : positions) (untagged : list variable)
    (tagged : tags) (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map
    (fun '(k, h) =>
       (k,
        untag_handler sites active untagged tagged (positions_of sites k) h))
    handlers.

Lemma untag_handlers_bind_statics (sites active : positions)
    (untagged : list variable) (tagged : tags)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics (untag_handlers sites active untagged tagged handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics, untag_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map existsb]. rewrite untag_handler_binds_statics, IH. reflexivity.
Qed.

Lemma lookup_untag_handlers (sites active : positions)
    (untagged : list variable) (tagged : tags) (k1 : continuation)
    (handlers : list (continuation * continuation_handler)) :
  lookup k1 (untag_handlers sites active untagged tagged handlers)
  = option_map
      (untag_handler sites active untagged tagged (positions_of sites k1))
      (lookup k1 handlers).
Proof.
  unfold untag_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map]. rewrite !lookup_cons, IH.
  destruct (Pos.eqb_spec k1 k') as [-> | Hne]; reflexivity.
Qed.

Lemma positions_of_cons (active : positions) (k k1 : continuation)
    (us : list nat) :
  positions_of ((k, us) :: active) k1
  = if Pos.eqb k1 k then us else positions_of active k1.
Proof.
  unfold positions_of. rewrite lookup_cons.
  destruct (Pos.eqb k1 k); reflexivity.
Qed.

Lemma positions_of_active_under (sites active : positions)
    (handlers : list (continuation * continuation_handler))
    (k1 : continuation) :
  positions_of (active_under sites handlers active) k1
  = match lookup k1 handlers with
    | Some _ => positions_of sites k1
    | None => positions_of active k1
    end.
Proof.
  unfold positions_of at 1, active_under. rewrite lookup_app.
  assert (Hlookup :
    lookup k1 (map (fun '(k, _) => (k, positions_of sites k)) handlers)
    = option_map (fun _ => positions_of sites k1) (lookup k1 handlers)).
  { induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
    cbn [map]. rewrite !lookup_cons, IH.
    destruct (Pos.eqb_spec k1 k') as [-> | Hne]; reflexivity. }
  rewrite Hlookup. destruct (lookup k1 handlers); reflexivity.
Qed.

Lemma find_arm_untag (active : positions) (tagged : tags) (j : Z)
    (arms : list (Z * apply_cont)) :
  find_arm j
    (map (fun '(n, ac) => (n, untag_apply_cont active tagged ac)) arms)
  = option_map (untag_apply_cont active tagged) (find_arm j arms).
Proof.
  induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
  cbn [map find_arm]. destruct (Z.eqb j n); [ reflexivity | exact IH ].
Qed.

(* What a defining expression that untags a variable is. *)
Lemma untag_of_some (d : named) (p : variable) :
  untag_of d = Some p -> d = Prim (Unary Untag_immediate) [Var p].
Proof.
  destruct d as [s | [[] | | | | ] [ | [x | sym | c] [ | a args]] | | | | | ];
    simpl; try discriminate.
  intros [= ->]. reflexivity.
Qed.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

Section Runs.

Variable sites : positions.

Definition handlers_ok (active : positions) (untagged : list variable)
    (tagged : tags) (handlers : list (continuation * continuation_handler))
    : bool :=
  forallb
    (fun '(k, h) =>
       handler_ok sites active untagged tagged (positions_of sites k) h)
    handlers.

(* What a continuation name stands for in the two runs, and which of
   the values that it is given are untagged in the second: the same
   thing; or two closures of one handler under the same variables and
   continuations so related; or a closure of a handler that the test
   accepts and a closure of that handler rewritten, under variables and
   continuations that are related; or two so related with the same
   extra arguments; or what is left of the same over-application. *)
Inductive cont_rel : list nat -> cont_value -> cont_value -> Prop :=
  | Rel_same (c : cont_value) : cont_rel [] c c
  | Rel_plain (hh : continuation_handler) (dv : env) (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    (∀ k1 : continuation, cont_rel [] (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel [] (Closure hh dv dc recursive) (Closure hh dv dc' recursive)
  | Rel_inside (us : list nat) (hh : continuation_handler) (dv dv' : env)
      (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler))
      (active : positions) (untagged : list variable) (tagged : tags) :
    handler_ok sites (active_under sites recursive active) untagged tagged us
      hh
    = true ->
    handlers_ok (active_under sites recursive active) untagged tagged
      recursive
    = true ->
    vars_rel untagged dv dv' ->
    tags_ok untagged tagged dv ->
    (∀ k1 : continuation,
      cont_rel (positions_of active k1) (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel us (Closure hh dv dc recursive)
      (Closure
         (untag_handler sites (active_under sites recursive active) untagged
            tagged us hh)
         dv' dc'
         (untag_handlers sites (active_under sites recursive active) untagged
            tagged recursive))
  | Rel_extra (c c' : cont_value) (extra : list value) :
    cont_rel [] c c' -> cont_rel [] (Extra_args c extra) (Extra_args c' extra)
  | Rel_over (remaining : list value) (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    cont_rel [] return_to return_to' ->
    cont_rel [] exn_to exn_to' ->
    cont_rel []
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Definition conts_rel (active : positions) (conts conts' : cont_env) : Prop :=
  ∀ k1 : continuation,
    cont_rel (positions_of active k1) (resolve k1 conts) (resolve k1 conts').

(* Two states: the same expression, under the same variables; or an
   expression that the test accepts and the same rewritten. *)
Inductive state_rel : state -> state -> Prop :=
  | Rel_state_plain (e : expr) (vars : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) :
    conts_rel [] conts conts' ->
    state_rel (State e vars conts statics hp rd)
      (State e vars conts' statics hp rd)
  | Rel_state_inside (e : expr) (vars vars' : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) (active : positions)
      (untagged : list variable) (tagged : tags) :
    untag_ok sites active untagged tagged e = true ->
    vars_rel untagged vars vars' ->
    tags_ok untagged tagged vars ->
    conts_rel active conts conts' ->
    state_rel (State e vars conts statics hp rd)
      (State (untag_expr sites active untagged tagged e) vars' conts' statics
         hp rd).

Lemma state_rel_refl (s : state) : state_rel s s.
Proof.
  destruct s as [e vars conts statics hp rd].
  apply Rel_state_plain. intros k1. apply Rel_same.
Qed.

Lemma simulated_of_related (r r' : step_result) :
  result_related state_rel r r' -> result_simulated state_rel r r'.
Proof.
  destruct r as [t | k1 vs hp rd | ], r' as [t' | k1' vs' hp' rd' | ]; simpl;
    try contradiction; auto.
Qed.

Lemma with_extra_args_rel (c c' : cont_value) (extra : list value) :
  cont_rel [] c c' ->
  cont_rel [] (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; simpl; [ exact Hc | apply Rel_extra, Hc ].
Qed.

(* The closures of the handlers of a recursive [Let_cont]. *)
Lemma rec_conts_plain
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts conts' : cont_env) :
  conts_rel [] conts conts' ->
  conts_rel [] (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers vars conts' ++ conts').
Proof.
  intros Hconts k1. rewrite !resolve_rec_closures.
  destruct (lookup k1 handlers) as [hh | ]; [ | apply Hconts ].
  apply Rel_plain. exact Hconts.
Qed.

Lemma rec_conts_inside
    (handlers : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) (active : positions)
    (untagged : list variable) (tagged : tags) :
  handlers_ok (active_under sites handlers active) untagged tagged handlers
  = true ->
  vars_rel untagged vars vars' ->
  tags_ok untagged tagged vars ->
  conts_rel active conts conts' ->
  conts_rel (active_under sites handlers active)
    (rec_closures handlers vars conts ++ conts)
    (rec_closures
       (untag_handlers sites (active_under sites handlers active) untagged
          tagged handlers)
       vars' conts'
     ++ conts').
Proof.
  intros Hok Hvars Htags Hconts k1.
  rewrite !resolve_rec_closures, lookup_untag_handlers,
    positions_of_active_under.
  destruct (lookup k1 handlers) as [hh | ] eqn:Hlookup; cbn [option_map];
    [ | exact (Hconts k1) ].
  apply Rel_inside; try assumption.
  unfold handlers_ok in Hok. rewrite forallb_forall in Hok.
  exact (Hok (k1, hh) (lookup_in k1 hh handlers Hlookup)).
Qed.

(* The same full call, returning and raising to related continuations. *)
Lemma call_code_rel (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel [] return_to return_to' ->
  cont_rel [] exn_to exn_to' ->
  result_related state_rel
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c vs args_arity return_arity return_to' exn_to'
       statics hp rd).
Proof.
  intros Hreturn Hraise.
  destruct (call_code_cases closure c vs args_arity return_arity statics hp rd)
    as [Hundefined | (callee_vars & Hcall)].
  - rewrite !Hundefined. exact I.
  - rewrite !Hcall. simpl.
    apply Rel_state_plain.
    intros k'. cbn [positions_of lookup].
    destruct (Pos.eq_dec k' (code_return_continuation c)) as [-> | Hne];
      [ rewrite !resolve_head; exact Hreturn | ].
    rewrite !(resolve_other k' (code_return_continuation c)) by exact Hne.
    destruct (Pos.eq_dec k' (code_exn_continuation c)) as [-> | Hne'];
      [ rewrite !resolve_head; exact Hraise | ].
    rewrite !(resolve_other k' (code_exn_continuation c)) by exact Hne'.
    apply Rel_same.
Qed.

Lemma run_call_rel (shape : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel [] return_to return_to' ->
  cont_rel [] exn_to exn_to' ->
  (∀ partial : value,
    result_simulated state_rel
      (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value return_to' [partial] statics hp rd)) ->
  result_simulated state_rel
    (run_call
       (fun partial => apply_cont_value return_to [partial] statics hp rd)
       shape return_arity return_to exn_to statics hp rd)
    (run_call
       (fun partial => apply_cont_value return_to' [partial] statics hp rd)
       shape return_arity return_to' exn_to' statics hp rd).
Proof.
  intros Hreturn Hraise Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply simulated_of_related, call_code_rel; assumption.
  - apply simulated_of_related, call_code_rel;
      [ apply Rel_over; assumption | exact Hraise ].
  - apply Hpartial.
  - exact I.
Qed.

(* Related continuations, applied to values of which those at the
   untagged positions are tagged immediates, and to the same with those
   untagged. *)
Lemma apply_cont_value_rel (us : list nat) (c c' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel us c c' ->
  ∀ vs : list value,
    Forall is_tagged (moved_from us 0 vs) ->
    result_simulated state_rel (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' (untag_values us vs) statics hp rd).
Proof.
  intros Hc.
  induction Hc
    as [c
       | [ps m he] dv dc dc' recursive Hdc _
       | us [ps m he] dv dv' dc dc' recursive active untagged tagged Hok
           Hrec Hvars Htags Hdc _
       | c c' extra Hc IHc
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise _];
    intros vs Htagged.
  - rewrite untag_values_nil.
    apply simulated_of_related, result_related_refl, state_rel_refl.
  - rewrite untag_values_nil. simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    destruct (bind_params ps vs dv) as [bound | ]; [ | exact I ].
    apply Rel_state_plain, rec_conts_plain. exact Hdc.
  - cbn [untag_handler apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m))
      eqn:Hkinds; [ | exact I ].
    destruct (bind_params ps vs dv) as [bound | ] eqn:Hbound; [ | exact I ].
    cbn [handler_ok] in Hok. apply andb_true_iff in Hok as [Hfits Hok].
    unfold handler_fits in Hfits.
    apply andb_true_iff in Hfits as [Hdistinct _].
    rewrite (kinds_untagged us vs m Htagged Hkinds).
    destruct
      (bind_params_rel untagged us ps vs dv dv' bound Hdistinct Hvars Htagged
         Hbound)
      as (bound' & Hbound' & Hrel).
    rewrite Hbound'.
    apply Rel_state_inside; [ exact Hok | exact Hrel | | ].
    + exact
        (tags_ok_params untagged tagged us ps vs dv bound Hdistinct Htags
           Hbound).
    + apply rec_conts_inside; assumption.
  - rewrite untag_values_nil. cbn [apply_cont_value].
    specialize (IHc (vs ++ extra)). rewrite untag_values_nil in IHc.
    apply IHc. rewrite moved_from_nil. constructor.
  - rewrite untag_values_nil, !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; try exact I.
    apply run_call_rel; [ exact Hreturn | exact Hraise | ].
    intros partial. specialize (IHreturn [partial]).
    rewrite untag_values_nil in IHreturn. apply IHreturn.
    rewrite moved_from_nil. constructor.
Qed.

Lemma apply_cont_value_plain (c c' : cont_value) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel [] c c' ->
  result_simulated state_rel (apply_cont_value c vs statics hp rd)
    (apply_cont_value c' vs statics hp rd).
Proof.
  intros Hc.
  pose proof (apply_cont_value_rel [] c c' statics hp rd Hc vs) as Happly.
  rewrite untag_values_nil, moved_from_nil in Happly.
  apply Happly. constructor.
Qed.

Lemma call_function_rel (callee : value) (call_kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel [] return_to return_to' ->
  cont_rel [] exn_to exn_to' ->
  result_simulated state_rel
    (call_function callee call_kind vs args_arity return_arity return_to
       exn_to statics hp rd)
    (call_function callee call_kind vs args_arity return_arity return_to'
       exn_to' statics hp rd).
Proof.
  intros Hreturn Hraise. unfold call_function.
  apply run_call_rel; [ exact Hreturn | exact Hraise | ].
  intros partial. apply apply_cont_value_plain, Hreturn.
Qed.

(* The untagging of an untagged variable, and the variable itself. *)
Lemma eval_untagging (untagged : list variable) (statics : static_env)
    (vars vars' : env) (p : variable) (v : value) :
  vars_rel untagged vars vars' ->
  among p untagged = true ->
  eval_named statics vars (Prim (Unary Untag_immediate) [Var p]) = Some v ->
  eval_named statics vars' (Simple (Var p)) = Some v.
Proof.
  intros Hvars Hp. pose proof (Hvars p) as Hrel. rewrite Hp in Hrel.
  destruct Hrel as (i & Hleft & Hright).
  cbn [eval_named eval_simples eval_simple]. rewrite Hleft, Hright.
  simpl. rewrite eval_prim_unary. simpl.
  destruct (in_range _ i); [ | discriminate ].
  intros [= <-]. reflexivity.
Qed.

(* Two related states take a step each, to related states or to the
   same exit, unless the first is undefined. *)
Lemma untag_step (s s' : state) :
  state_rel s s' -> result_simulated state_rel (step s) (step s').
Proof.
  intros
    [e vars conts conts' statics hp rd Hconts
    | e vars vars' conts conts' statics hp rd active untagged tagged Hok
        Hvars Htags Hconts].
  - (* the same expression *)
    destruct e as [y kind1 d body | bindings body | k1 hh body
                  | handlers body | ap | ac | scrutinee arms | ].
    + rewrite !step_let_exec.
      destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
        [ | exact I ].
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      apply Rel_state_plain, Hconts.
    + rewrite !step_let_static.
      destruct (eval_static_bindings vars bindings statics); [ | exact I ].
      apply Rel_state_plain, Hconts.
    + rewrite !step_let_cont_non_recursive.
      apply Rel_state_plain.
      intros k2. destruct (Pos.eq_dec k2 k1) as [-> | Hne].
      * rewrite !resolve_head. apply Rel_plain. exact Hconts.
      * rewrite !(resolve_other k2 k1) by exact Hne. apply Hconts.
    + rewrite !step_let_cont_recursive.
      destruct (handlers_bind_statics handlers); [ exact I | ].
      apply Rel_state_plain, rec_conts_plain, Hconts.
    + rewrite !step_apply.
      destruct (eval_simple vars (ap_callee ap)) as [callee | ];
        [ | exact I ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
        [ | exact I ].
      apply call_function_rel;
        [ apply Hconts | apply with_extra_args_rel, Hconts ].
    + rewrite !step_apply_cont.
      destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
      unfold apply_continuation. apply apply_cont_value_plain, Hconts.
    + rewrite !step_switch.
      destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
      destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
      destruct (find_arm j arms) as [ac | ]; [ | exact I ].
      apply Rel_state_plain, Hconts.
    + rewrite !step_invalid. exact I.
  - (* an expression and the same rewritten *)
    destruct e as [y kind1 d body | bindings body | k1 hh body
                  | handlers body | ap | ac | scrutinee arms | ];
      cbn [untag_expr untag_ok] in Hok |- *.
    + apply andb_true_iff in Hok as [Hd Hbody].
      rewrite !step_let_exec.
      destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ]
        eqn:Hv; [ | exact I ].
      assert
        (Hv' :
          exec_named statics hp rd vars' (untag_named untagged d)
          = Some (v, hp', rd')).
      { unfold named_ok in Hd. unfold untag_named.
        destruct (reads_untagged untagged d) eqn:Hreads.
        - unfold reads_untagged in Hreads.
          destruct (untag_of d) as [p | ] eqn:Huntag; [ | discriminate ].
          rewrite Hreads. apply untag_of_some in Huntag. subst d.
          (* the untagging reads no cell, and the variable is worth
             what it gave *)
          apply exec_named_pure in Hv as (Hv & -> & ->); [ | reflexivity ].
          cbn [exec_named].
          rewrite
            (eval_untagging untagged statics vars vars' p v Hvars Hreads Hv).
          reflexivity.
        - cbn [orb] in Hd.
          assert (Hsame :
            exec_named statics hp rd vars' d
            = exec_named statics hp rd vars d)
            by exact
              (exec_named_plain untagged statics hp rd vars vars' d Hvars
                 Hd).
          unfold reads_untagged in Hreads.
          destruct (untag_of d) as [p | ]; [ rewrite Hreads | ];
            rewrite Hsame; exact Hv. }
      rewrite Hv'.
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      apply Rel_state_inside;
        [ exact Hbody | apply vars_rel_cons, Hvars | | exact Hconts ].
      exact
        (tags_ok_let untagged tagged statics hp hp' rd rd' vars y d v Hd
           Htags Hv).
    + apply andb_true_iff in Hok as [Hbindings Hbody].
      rewrite !step_let_static,
        (eval_static_bindings_plain untagged vars vars' bindings statics Hvars
           Hbindings).
      destruct (eval_static_bindings vars bindings statics); [ | exact I ].
      apply Rel_state_inside; assumption.
    + apply andb_true_iff in Hok as [Hhh Hbody].
      rewrite !step_let_cont_non_recursive.
      apply Rel_state_inside; [ exact Hbody | exact Hvars | exact Htags | ].
      intros k2. rewrite positions_of_cons.
      destruct (Pos.eqb_spec k2 k1) as [-> | Hne].
      * rewrite !resolve_head.
        exact
          (Rel_inside (positions_of sites k1) hh vars vars' conts conts' []
             active untagged tagged Hhh eq_refl Hvars Htags Hconts).
      * rewrite !(resolve_other k2 k1) by exact Hne. exact (Hconts k2).
    + apply andb_true_iff in Hok as [Hhandlers Hbody].
      rewrite !step_let_cont_recursive.
      fold
        (untag_handlers sites (active_under sites handlers active) untagged
           tagged handlers).
      rewrite untag_handlers_bind_statics.
      destruct (handlers_bind_statics handlers); [ exact I | ].
      apply Rel_state_inside; [ exact Hbody | exact Hvars | exact Htags | ].
      apply rec_conts_inside; assumption.
    + apply andb_true_iff in Hok as [Hok Hraise].
      apply andb_true_iff in Hok as [Hargs Hreturn].
      assert (Hcallee :
        none_in untagged (fun p => var_in_simple p (ap_callee ap)) = true).
      { unfold none_in in Hargs |- *. rewrite forallb_forall in Hargs.
        apply forallb_forall. intros p Hp. specialize (Hargs p Hp).
        apply negb_true_iff in Hargs. apply orb_false_iff in Hargs as
          [Hargs _].
        apply orb_false_iff in Hargs as [Hargs _]. rewrite Hargs.
        reflexivity. }
      assert (Hap_args :
        none_in untagged (fun p => var_in_simples p (ap_args ap)) = true).
      { unfold none_in in Hargs |- *. rewrite forallb_forall in Hargs.
        apply forallb_forall. intros p Hp. specialize (Hargs p Hp).
        apply negb_true_iff in Hargs. apply orb_false_iff in Hargs as
          [Hargs _].
        apply orb_false_iff in Hargs as [_ Hargs]. rewrite Hargs.
        reflexivity. }
      assert (Hextra :
        none_in untagged (fun p => var_in_simples p (exn_extra_simples ap))
        = true).
      { unfold none_in in Hargs |- *. rewrite forallb_forall in Hargs.
        apply forallb_forall. intros p Hp. specialize (Hargs p Hp).
        apply negb_true_iff in Hargs. apply orb_false_iff in Hargs as
          [_ Hargs].
        rewrite Hargs. reflexivity. }
      rewrite !step_apply,
        (eval_simple_plain untagged vars vars' _ Hvars Hcallee),
        (eval_simples_plain untagged vars vars' _ Hvars Hap_args),
        (eval_simples_plain untagged vars vars' _ Hvars Hextra).
      destruct (eval_simple vars (ap_callee ap)) as [callee | ];
        [ | exact I ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
        [ | exact I ].
      pose proof (Hconts (ap_continuation ap)) as Hto.
      pose proof (Hconts (ap_exn_continuation ap)) as Hexn.
      unfold no_positions in Hreturn, Hraise.
      destruct (positions_of active (ap_continuation ap));
        [ | discriminate Hreturn ].
      destruct (positions_of active (ap_exn_continuation ap));
        [ | discriminate Hraise ].
      apply call_function_rel;
        [ exact Hto | apply with_extra_args_rel, Hexn ].
    + rewrite !step_apply_cont. unfold untag_apply_cont.
      cbn [ac_args ac_continuation].
      destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
        [ | exact I ].
      unfold apply_cont_ok in Hok.
      destruct
        (eval_untag_args untagged tagged _ vars vars' (ac_args ac) vs Hvars
           Htags Hok Hvs)
        as [Hvs' Htagged].
      rewrite Hvs'. unfold apply_continuation.
      exact
        (apply_cont_value_rel _ _ _ statics hp rd
           (Hconts (ac_continuation ac)) vs Htagged).
    + apply andb_true_iff in Hok as [Hscrutinee Harms].
      rewrite !step_switch,
        (eval_simple_plain untagged vars vars' _ Hvars Hscrutinee).
      destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
      destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
      rewrite find_arm_untag.
      destruct (find_arm j arms) as [ac | ] eqn:Hfind; [ | exact I ].
      cbn [option_map].
      rewrite forallb_forall in Harms.
      exact
        (Rel_state_inside (Apply_cont ac) vars vars' conts conts' statics hp
           rd active untagged tagged
           (Harms (j, ac) (find_arm_in j arms ac Hfind)) Hvars Htags Hconts).
    + rewrite !step_invalid. exact I.
Qed.

Lemma related_refines (s s' : state) : state_rel s s' -> refines s s'.
Proof. exact (simulation_refines state_rel untag_step s s'). Qed.

End Runs.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Theorem untag_at_refines : untag_at_refines_statement.
Proof.
  intros sites e vars conts statics hp rd. unfold untag_at.
  destruct (untag_ok sites [] [] [] e) eqn:Hok; [ | apply refines_refl ].
  apply (related_refines sites).
  apply Rel_state_inside;
    [ exact Hok | apply vars_rel_nil | apply tags_ok_nil | ].
  intros k1. apply Rel_same.
Qed.

(* The pass of the validation is the rewrite at the positions that it
   computes. *)
Lemma untag_params_refines (e : expr) : expr_refines e (untag_params e).
Proof. apply untag_at_refines. Qed.
