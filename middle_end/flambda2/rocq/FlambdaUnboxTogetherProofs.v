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


(* Proofs of the statement of FlambdaUnboxTogetherSpec.v. Nothing here
   needs review.

   For one sort of box, the term and its result run in step under a
   relation of states ([state_rel]), as in
   FlambdaUntagParamsProofs.v: the same expression, where neither run
   is in a term that the pass rewrote; or an expression that the test
   accepts and the same rewritten, under variables that agree but for
   the unboxed ones, and under continuations that are closures of a
   handler and of that handler rewritten. An unboxed variable holds,
   in the second run, a value of the kind of the number that is what
   [Unbox_number] has on what it holds in the first, if it has a
   value, whatever is bound later ([unboxes_to]): the first run may
   hold a box or the symbol of a static one. The unboxing reads no
   cell, so that is said of its value alone, and no heap is named.
   What is known holds of the variables of the first run, at its heap
   and its record ([facts_hold]), which is what says that an argument
   that the term shows to be a box is given its number. That the
   result refines the term is then
   FlambdaSemanticsProofs.simulation_refines. *)

From Stdlib Require Import ZArith List Bool Arith Lia Permutation.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaPrimitiveModel FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaHeapSpec
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaUnusedParamsProofs FlambdaBlockLoadSpec FlambdaUnboxSpec
  FlambdaUnboxParamsSpec FlambdaUnboxParamsProofs FlambdaUnboxVariantsSpec
  FlambdaUnboxVariantsProofs FlambdaUnboxInScopeSpec
  FlambdaUnboxInScopeProofs FlambdaUntagParamsSpec FlambdaUnboxTogetherSpec.
From OxCamlFlambda2 Require FlambdaUntagParamsProofs.
Import ListNotations.

Module Untag := FlambdaUntagParamsProofs.

(* ======================================================================== *)
(* The sort of a box, and what the term shows                               *)
(* ======================================================================== *)

Lemma unboxed_kind_injective (boxed boxed' : boxable_number) :
  kind_eqb (unboxed_kind boxed') (unboxed_kind boxed) = true ->
  boxed' = boxed.
Proof.
  destruct boxed', boxed; intros H;
    first [ reflexivity | cbv in H; discriminate H ].
Qed.

(* What a defining expression that unboxes a variable is. *)
Lemma unbox_of_some (boxed : boxable_number) (d : named) (p : variable) :
  unbox_of boxed d = Some p -> d = Prim (Unary (Unbox_number boxed)) [Var p].
Proof.
  destruct d as [s | [[] | | | | ] [ | [x | sym | c] [ | a args]] | | | | | ];
    cbn [unbox_of]; try discriminate.
  match goal with
  | |- context [kind_eqb (unboxed_kind ?b) (unboxed_kind boxed)] =>
    destruct (kind_eqb (unboxed_kind b) (unboxed_kind boxed)) eqn:Hkind;
      [ | discriminate ];
      intros [= ->]; rewrite (unboxed_kind_injective boxed b Hkind);
      reflexivity
  end.
Qed.

(* What [certain_projection] gives is of the kind, and is what the
   primitive gives of the argument, if it gives a value, whatever is
   bound later and wherever the heap and the record evolve to. *)
Lemma certain_projection_gives (known : facts) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record)
    (p : unary_primitive) (kind : flambda_kind) (arg projection : simple)
    (v : value) :
  facts_hold known vars statics hp rd ->
  certain_projection known p kind arg = Some projection ->
  eval_simple vars arg = Some v ->
  ∃ u : value,
    eval_simple vars projection = Some u
    ∧ kind_of_value u = kind
    ∧ gives p v u statics hp rd.
Proof.
  intros Hfacts Hcertain Harg. unfold certain_projection in Hcertain.
  assert (Hmade :
    match made_field known p kind arg with
    | Some contents => Some contents
    | None => made_number known p kind arg
    end
    = Some projection ->
    ∃ u : value,
      eval_simple vars projection = Some u
      ∧ kind_of_value u = kind
      ∧ gives p v u statics hp rd).
  { destruct (made_field known p kind arg) as [contents | ] eqn:Hfield.
    - intros [= <-].
      exact (made_field_gives known vars statics hp rd p kind arg contents v
        Hfacts Hfield Harg).
    - intros Hnumber.
      exact (made_number_gives known vars statics hp rd p kind arg
        projection v Hfacts Hnumber Harg). }
  destruct (projection_of known p arg) as [[y | sym | c] | ] eqn:Hknown;
    try exact (Hmade Hcertain).
  destruct (kind_eqb (kind_of_const c) kind) eqn:Hkind; [ | discriminate ].
  injection Hcertain as <-. exists (Value_const c).
  split; [ reflexivity | split; [ apply kind_eqb_eq, Hkind | ] ].
  intros statics' hp' rd' Hmore Hevolves w hp'' Hprim.
  pose proof (projection_of_sound known vars statics' hp' hp'' rd' p arg
    (Const c) v w
    (facts_hold_evolves known vars statics' hp hp' rd rd'
       (facts_hold_extend known vars statics statics' hp rd Hfacts Hmore)
       Hevolves)
    Hknown Harg Hprim) as Hw.
  simpl in Hw. congruence.
Qed.

(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

(* With no position, the values are as they were. *)
Lemma plain_values {R : value -> value -> Prop} (vs ws : list value) :
  Forall2 R (moved_from [] 0 vs) ws -> kept_from [] 0 vs ++ rev ws = vs.
Proof.
  rewrite Untag.moved_from_nil. intros H. inversion H.
  rewrite Untag.kept_from_nil. apply app_nil_r.
Qed.

(* The parameters and the values in the order of the rewrite: a
   parameter that is moved has the value given for it, and each other
   the value that it had. *)
Lemma lookup_combine_moved_rel (us : list nat) (x : variable)
    (params : list variable) :
  ∀ (vs ws : list value) (i : nat),
    distinct params = true ->
    length params = length vs ->
    length ws = length (moved_from us i vs) ->
    lookup x
      (combine (kept_from us i params ++ rev (moved_from us i params))
         (kept_from us i vs ++ rev ws))
    = if among x (moved_from us i params)
      then lookup x (combine (moved_from us i params) ws)
      else lookup x (combine params vs).
Proof.
  induction params as [ | p params IH]; intros [ | v vs] ws i Hdistinct
    Hlength Hws; try discriminate Hlength.
  - cbn [kept_from moved_from] in Hws |- *.
    destruct ws; [ reflexivity | discriminate Hws ].
  - cbn [distinct] in Hdistinct.
    apply andb_true_iff in Hdistinct as [Hp Hdistinct].
    apply negb_true_iff in Hp. injection Hlength as Hlength.
    cbn [kept_from moved_from] in Hws |- *.
    destruct (at_position us i).
    + destruct ws as [ | w ws]; [ discriminate Hws | ].
      injection Hws as Hws.
      specialize (IH vs ws (S i) Hdistinct Hlength Hws).
      cbn [rev]. rewrite !app_assoc, Untag.combine_snoc, lookup_app, IH.
      2: { rewrite !length_app, !length_rev.
           rewrite (Untag.kept_from_length us params vs (S i) Hlength), Hws,
             (Untag.moved_from_length us params vs (S i) Hlength).
           reflexivity. }
      rewrite among_cons. cbn [combine]. rewrite !lookup_cons.
      cbn [lookup].
      destruct (Pos.eqb_spec x p) as [-> | Hne]; cbn [orb].
      * destruct (among p (moved_from us (S i) params)) eqn:Hmoved.
        -- apply Untag.among_moved_from in Hmoved. congruence.
        -- rewrite (Untag.lookup_combine_none p params vs Hp). reflexivity.
      * destruct (among x (moved_from us (S i) params));
          match goal with
          | |- match ?found with _ => _ end = _ => destruct found
          end; reflexivity.
    + specialize (IH vs ws (S i) Hdistinct Hlength Hws).
      cbn [app combine]. rewrite !lookup_cons.
      destruct (Pos.eqb_spec x p) as [-> | Hne]; [ | exact IH ].
      destruct (among p (moved_from us (S i) params)) eqn:Hmoved;
        [ | reflexivity ].
      apply Untag.among_moved_from in Hmoved. congruence.
Qed.

(* The value that a parameter moved had and the value given for it are
   related as those moved are. *)
Lemma moved_values_related (R : value -> value -> Prop) (us : list nat)
    (x : variable) (params : list variable) :
  ∀ (vs ws : list value) (i : nat),
    distinct params = true ->
    length params = length vs ->
    Forall2 R (moved_from us i vs) ws ->
    among x (moved_from us i params) = true ->
    ∃ v u : value,
      lookup x (combine params vs) = Some v
      ∧ lookup x (combine (moved_from us i params) ws) = Some u
      ∧ R v u.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] ws i Hdistinct
    Hlength HR Hx; try discriminate Hlength.
  - cbn [moved_from] in Hx. discriminate Hx.
  - cbn [distinct] in Hdistinct.
    apply andb_true_iff in Hdistinct as [Hp Hdistinct].
    apply negb_true_iff in Hp. injection Hlength as Hlength.
    cbn [moved_from combine] in HR, Hx |- *. rewrite lookup_cons.
    destruct (at_position us i).
    + inversion HR as [ | v0 w rest ws' Hvw Hrest]; subst.
      rewrite among_cons in Hx. cbn [combine]. rewrite lookup_cons.
      destruct (Pos.eqb_spec x p) as [-> | Hne].
      * exists v, w. auto.
      * cbn [orb] in Hx. exact (IH vs ws' (S i) Hdistinct Hlength Hrest Hx).
    + destruct (Pos.eqb_spec x p) as [-> | Hne].
      * apply Untag.among_moved_from in Hx. congruence.
      * exact (IH vs ws (S i) Hdistinct Hlength HR Hx).
Qed.

Section Sort.

Variable boxed : boxable_number.

(* ======================================================================== *)
(* Values and variables                                                     *)
(* ======================================================================== *)

(* What the second run holds for a box of the first: a value of the
   kind of the number, which is what the unboxing has on the box, if
   it has a value, whatever is bound later. The unboxing reads no
   cell: what it has on a value is what it gives of it on any heap. *)
Definition unboxes_to (statics : static_env) (v u : value) : Prop :=
  (∀ statics' : static_env,
    statics_extend statics statics' ->
    ∀ w : value,
      eval_prim statics' (Unary (Unbox_number boxed)) [v] = Some w -> w = u)
  ∧ kind_of_value u = unboxed_kind boxed.

Lemma unboxes_to_extend (statics statics' : static_env) (v u : value) :
  unboxes_to statics v u ->
  statics_extend statics statics' ->
  unboxes_to statics' v u.
Proof.
  intros [Hgives Hkind] Hmore. split; [ | exact Hkind ].
  intros statics'' Hmore'.
  exact (Hgives statics'' (statics_extend_trans _ _ _ Hmore Hmore')).
Qed.

(* What the unboxing gives of a value on a heap, it has on the value:
   it has no value on a location. *)
Lemma gives_unboxes_to (statics : static_env) (hp : heap) (rd : read_record)
    (v u : value) :
  gives (Unbox_number boxed) v u statics hp rd ->
  kind_of_value u = unboxed_kind boxed ->
  unboxes_to statics v u.
Proof.
  intros Hgives Hkind. split; [ | exact Hkind ].
  intros statics' Hmore w Hprim.
  apply (Hgives statics' hp rd Hmore (memory_evolves_refl hp rd) w hp).
  apply eval_prim_exec_prim; [ reflexivity | | exact Hprim ].
  intros l [-> | []]. rewrite eval_prim_unary in Hprim. discriminate Hprim.
Qed.

(* The variables of the two runs: an unboxed one holds a box in the
   first and its number in the second; the others hold the same. *)
Definition vars_rel (statics : static_env) (unboxed : list variable)
    (vars vars' : env) : Prop :=
  ∀ x : variable,
    if among x unboxed
    then
      ∃ v u : value,
        lookup x vars = Some v
        ∧ lookup x vars' = Some u
        ∧ unboxes_to statics v u
    else lookup x vars = lookup x vars'.

Lemma vars_rel_nil (statics : static_env) (vars : env) :
  vars_rel statics [] vars vars.
Proof. intros x. reflexivity. Qed.

Lemma vars_rel_cons (statics : static_env) (unboxed : list variable)
    (vars vars' : env) (y : variable) (v : value) :
  vars_rel statics unboxed vars vars' ->
  vars_rel statics (without [y] unboxed) ((y, v) :: vars) ((y, v) :: vars').
Proof.
  intros Hvars x. rewrite among_without, among_cons, !lookup_cons.
  cbn [among existsb]. rewrite orb_false_r.
  destruct (Pos.eqb x y) eqn:Hxy.
  - rewrite andb_false_r. reflexivity.
  - rewrite andb_true_r. exact (Hvars x).
Qed.

Lemma vars_rel_extend (statics statics' : static_env)
    (unboxed : list variable) (vars vars' : env) :
  vars_rel statics unboxed vars vars' ->
  statics_extend statics statics' ->
  vars_rel statics' unboxed vars vars'.
Proof.
  intros Hvars Hmore x. specialize (Hvars x).
  destruct (among x unboxed); [ | exact Hvars ].
  destruct Hvars as (v & u & Hv & Hu & Hto). exists v, u.
  split; [ exact Hv | split; [ exact Hu | ] ].
  exact (unboxes_to_extend statics statics' v u Hto Hmore).
Qed.

Lemma eval_simple_plain (statics : static_env) (unboxed : list variable)
    (vars vars' : env) (s : simple) :
  vars_rel statics unboxed vars vars' ->
  none_in unboxed (fun p => var_in_simple p s) = true ->
  eval_simple vars' s = eval_simple vars s.
Proof.
  intros Hvars Hnone. apply eval_simple_agree. intros x Hx.
  pose proof (Hvars x) as Hx'.
  rewrite
    (Untag.none_in_not_among unboxed _ x Hnone (var_in_simple_sound x s Hx))
    in Hx'.
  exact Hx'.
Qed.

Lemma eval_simples_plain (statics : static_env) (unboxed : list variable)
    (vars vars' : env) (args : list simple) :
  vars_rel statics unboxed vars vars' ->
  none_in unboxed (fun p => var_in_simples p args) = true ->
  eval_simples vars' args = eval_simples vars args.
Proof.
  intros Hvars Hnone. apply eval_simples_agree. intros x Hx.
  pose proof (Hvars x) as Hx'.
  rewrite
    (Untag.none_in_not_among unboxed _ x Hnone
       (var_in_simples_sound x args Hx))
    in Hx'.
  exact Hx'.
Qed.

Lemma eval_named_plain (statics statics0 : static_env)
    (unboxed : list variable) (vars vars' : env) (d : named) :
  vars_rel statics0 unboxed vars vars' ->
  none_in unboxed (fun p => var_in_named p d) = true ->
  eval_named statics vars' d = eval_named statics vars d.
Proof.
  intros Hvars Hnone. apply eval_named_agree. intros x Hx.
  pose proof (Hvars x) as Hx'.
  rewrite
    (Untag.none_in_not_among unboxed _ x Hnone (var_in_named_sound x d Hx))
    in Hx'.
  exact Hx'.
Qed.

Lemma exec_named_plain (statics statics0 : static_env) (hp : heap)
    (rd : read_record) (unboxed : list variable) (vars vars' : env)
    (d : named) :
  vars_rel statics0 unboxed vars vars' ->
  none_in unboxed (fun p => var_in_named p d) = true ->
  exec_named statics hp rd vars' d = exec_named statics hp rd vars d.
Proof.
  intros Hvars Hnone. apply exec_named_agree. intros x Hx.
  pose proof (Hvars x) as Hx'.
  rewrite
    (Untag.none_in_not_among unboxed _ x Hnone (var_in_named_sound x d Hx))
    in Hx'.
  exact Hx'.
Qed.

Lemma eval_static_bindings_plain (statics0 : static_env)
    (unboxed : list variable) (vars vars' : env)
    (bindings : list static_binding) (statics : static_env) :
  vars_rel statics0 unboxed vars vars' ->
  none_in unboxed (fun p => existsb (var_in_static_binding p) bindings)
  = true ->
  eval_static_bindings vars' bindings statics
  = eval_static_bindings vars bindings statics.
Proof.
  intros Hvars Hnone. apply eval_static_bindings_agree. intros x Hx.
  pose proof (Hvars x) as Hx'.
  rewrite (Untag.none_in_not_among unboxed _ x Hnone) in Hx';
    [ exact Hx' | ].
  apply Exists_exists in Hx. destruct Hx as (binding & Hin & Hfree).
  apply existsb_exists. exists binding.
  split; [ exact Hin | apply var_in_static_binding_sound, Hfree ].
Qed.

(* The unboxing of an unboxed variable, and the variable itself. *)
Lemma eval_unboxing (statics : static_env) (unboxed : list variable)
    (vars vars' : env) (p : variable) (v : value) :
  vars_rel statics unboxed vars vars' ->
  among p unboxed = true ->
  eval_named statics vars (Prim (Unary (Unbox_number boxed)) [Var p])
  = Some v ->
  eval_named statics vars' (Simple (Var p)) = Some v.
Proof.
  intros Hvars Hp. pose proof (Hvars p) as Hrel. rewrite Hp in Hrel.
  destruct Hrel as (v0 & u & Hleft & Hright & Hgives & _).
  revert Hgives. generalize (Unbox_number boxed). intros q Hgives.
  simpl. rewrite Hleft, Hright. simpl. intros Hprim.
  rewrite (Hgives statics (statics_extend_refl statics) v Hprim).
  reflexivity.
Qed.

(* ======================================================================== *)
(* The arguments of an application                                          *)
(* ======================================================================== *)

Lemma eval_unboxed_simple (statics : static_env) (hp : heap)
    (rd : read_record) (unboxed : list variable) (known : facts)
    (vars vars' : env) (s s' : simple) (v : value) :
  vars_rel statics unboxed vars vars' ->
  facts_hold known vars statics hp rd ->
  unboxed_simple boxed unboxed known s = Some s' ->
  eval_simple vars s = Some v ->
  ∃ u : value, eval_simple vars' s' = Some u ∧ unboxes_to statics v u.
Proof.
  intros Hvars Hfacts Hs Hv.
  assert (Hshown :
    shown_number boxed unboxed known s = Some s' ->
    ∃ u : value, eval_simple vars' s' = Some u ∧ unboxes_to statics v u).
  { unfold shown_number.
    destruct
      (certain_projection known (Unbox_number boxed) (unboxed_kind boxed) s)
      as [number | ] eqn:Hcertain; [ | discriminate ].
    destruct (none_in unboxed (fun p => var_in_simple p number)) eqn:Hnone;
      [ | discriminate ].
    intros [= <-].
    destruct
      (certain_projection_gives known vars statics hp rd _ _ s number v
         Hfacts Hcertain Hv)
      as (u & Hu & Hkind & Hgives).
    exists u.
    split; [ | exact (gives_unboxes_to statics hp rd v u Hgives Hkind) ].
    rewrite (eval_simple_plain statics unboxed vars vars' number Hvars Hnone).
    exact Hu. }
  destruct s as [x | sym | c]; cbn [unboxed_simple] in Hs;
    try exact (Hshown Hs).
  destruct (among x unboxed) eqn:Hx; [ | exact (Hshown Hs) ].
  injection Hs as <-. pose proof (Hvars x) as Hrel. rewrite Hx in Hrel.
  destruct Hrel as (v0 & u & Hleft & Hright & Hto).
  simpl in Hv. rewrite Hleft in Hv. injection Hv as <-.
  exists u. split; [ simpl; exact Hright | exact Hto ].
Qed.

(* The arguments at the unboxed positions. *)
Lemma eval_simples_unboxed (statics : static_env) (hp : heap)
    (rd : read_record) (unboxed : list variable) (known : facts)
    (vars vars' : env) (args : list simple) :
  vars_rel statics unboxed vars vars' ->
  facts_hold known vars statics hp rd ->
  forallb (unboxed_arg boxed unboxed known) args = true ->
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    ∃ ws : list value,
      eval_simples vars' (map (unbox_arg boxed unboxed known) args) = Some ws
      ∧ Forall2 (unboxes_to statics) vs ws.
Proof.
  intros Hvars Hfacts.
  induction args as [ | s args IH]; intros Hok vs; simpl.
  - intros [= <-]. exists []. split; [ reflexivity | constructor ].
  - simpl in Hok. apply andb_true_iff in Hok as [Hs Hargs].
    destruct (eval_simple vars s) as [v | ] eqn:Hv; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ] eqn:Hrest;
      [ | discriminate ].
    intros [= <-]. destruct (IH Hargs rest eq_refl) as (ws & Heval & Hrel).
    unfold unboxed_arg in Hs.
    destruct (unboxed_simple boxed unboxed known s) as [s' | ] eqn:Hs';
      [ | discriminate ].
    assert (Harg : unbox_arg boxed unboxed known s = s')
      by (unfold unbox_arg; rewrite Hs'; reflexivity).
    destruct
      (eval_unboxed_simple statics hp rd unboxed known vars vars' s s' v
         Hvars Hfacts Hs' Hv)
      as (u & Hu & Hto).
    rewrite Harg, Hu, Heval. exists (u :: ws).
    split; [ reflexivity | constructor; assumption ].
Qed.

(* The arguments of an application that the test accepts. *)
Lemma eval_together_args (statics : static_env) (hp : heap)
    (rd : read_record) (unboxed : list variable) (known : facts)
    (us : list nat) (vars vars' : env) (args : list simple)
    (vs : list value) :
  vars_rel statics unboxed vars vars' ->
  facts_hold known vars statics hp rd ->
  together_args_ok boxed unboxed known us args = true ->
  eval_simples vars args = Some vs ->
  ∃ ws : list value,
    eval_simples vars' (together_args boxed us unboxed known args)
    = Some (kept_from us 0 vs ++ rev ws)
    ∧ Forall2 (unboxes_to statics) (moved_from us 0 vs) ws.
Proof.
  intros Hvars Hfacts Hok Hvs. unfold together_args_ok in Hok.
  apply andb_true_iff in Hok as [Hkept Hmoved].
  destruct (Untag.eval_simples_split us vars args 0 vs Hvs) as [Hk Hm].
  destruct
    (eval_simples_unboxed statics hp rd unboxed known vars vars' _ Hvars
       Hfacts Hmoved _ Hm)
    as (ws & Hm' & Hrel).
  exists ws. split; [ | exact Hrel ].
  unfold together_args.
  rewrite eval_simples_app,
    (eval_simples_plain statics unboxed vars vars' _ Hvars Hkept), Hk,
    (Untag.eval_simples_rev vars' _ _ Hm').
  reflexivity.
Qed.

(* ======================================================================== *)
(* The parameters of a handler                                              *)
(* ======================================================================== *)

(* The handler entered: the variables of the two runs are related
   again, with the parameters moved unboxed. *)
Lemma bind_params_rel (statics : static_env) (unboxed : list variable)
    (us : list nat) (params : list variable) (vs ws : list value)
    (vars vars' bound : env) :
  distinct params = true ->
  vars_rel statics unboxed vars vars' ->
  Forall2 (unboxes_to statics) (moved_from us 0 vs) ws ->
  bind_params params vs vars = Some bound ->
  ∃ bound' : env,
    bind_params (untag_params_list us params) (kept_from us 0 vs ++ rev ws)
      vars'
    = Some bound'
    ∧ vars_rel statics (untagged_under us params unboxed) bound bound'.
Proof.
  intros Hdistinct Hvars Hrel Hbind.
  pose proof (Untag.bind_params_same_length params vs vars bound Hbind)
    as Hlength.
  pose proof (Untag.distinct_untag_params_list us params Hdistinct)
    as Hdistinct'.
  pose proof (Forall2_length Hrel) as Hws.
  destruct
    (Untag.bind_params_of_length (untag_params_list us params)
       (kept_from us 0 vs ++ rev ws) vars')
    as [bound' Hbind'].
  { unfold untag_params_list.
    rewrite !length_app, !length_rev.
    rewrite (Untag.kept_from_length us params vs 0 Hlength), <- Hws,
      (Untag.moved_from_length us params vs 0 Hlength).
    reflexivity. }
  exists bound'. split; [ exact Hbind' | ].
  intros x.
  rewrite (Untag.lookup_bind_params params vs vars bound x Hdistinct Hbind),
    (Untag.lookup_bind_params _ _ vars' bound' x Hdistinct' Hbind').
  unfold untag_params_list.
  rewrite
    (lookup_combine_moved_rel us x params vs ws 0 Hdistinct Hlength
       (eq_sym Hws)).
  unfold untagged_under. rewrite among_app, among_without.
  destruct (among x (moved_from us 0 params)) eqn:Hmoved; cbn [orb].
  - destruct
      (moved_values_related (unboxes_to statics) us x params vs ws 0
         Hdistinct Hlength Hrel Hmoved)
      as (v & u & Hv & Hu & Hto).
    rewrite Hv, Hu. exists v, u. auto.
  - destruct (among x params) eqn:Hparam.
    + rewrite andb_false_r.
      destruct (Untag.lookup_combine_some x params vs Hlength Hparam)
        as [v Hv].
      rewrite Hv. reflexivity.
    + rewrite andb_true_r, (Untag.lookup_combine_none x params vs Hparam).
      exact (Hvars x).
Qed.

(* The kinds of the values in the order of the rewrite are those of the
   metadata rewritten. *)
Lemma kinds_unboxed (statics : static_env) (us : list nat)
    (vs ws : list value) (metadata : handler_metadata) :
  Forall2 (unboxes_to statics) (moved_from us 0 vs) ws ->
  kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata) = true ->
  kinds_eqb (map kind_of_value (kept_from us 0 vs ++ rev ws))
    (hm_params_kinds (together_metadata boxed us metadata))
  = true.
Proof.
  intros Hrel Hkinds. apply kinds_eqb_eq in Hkinds.
  unfold together_metadata. cbn [hm_params_kinds].
  rewrite <- Hkinds, map_app, Untag.kept_from_map, Untag.moved_from_map,
    !map_rev.
  assert (Hmoved :
    map kind_of_value ws
    = map (fun _ => unboxed_kind boxed)
        (map kind_of_value (moved_from us 0 vs))).
  { induction Hrel as [ | v u rest ws' [_ Hkind] _ IH]; [ reflexivity | ].
    cbn [map]. rewrite IH, Hkind. reflexivity. }
  rewrite Hmoved. apply kinds_eqb_refl.
Qed.

(* ======================================================================== *)
(* What the rewrite keeps of the syntax                                     *)
(* ======================================================================== *)

Fixpoint together_expr_binds_statics (sites active : positions)
    (unboxed : list variable) (known : facts) (e : expr) {struct e} :
  binds_statics (together_expr boxed sites active unboxed known e)
  = binds_statics e
with together_handler_binds_statics (sites active : positions)
    (unboxed : list variable) (known : facts) (us : list nat)
    (h : continuation_handler) {struct h} :
  handler_binds_statics (together_handler boxed sites active unboxed known us h)
  = handler_binds_statics h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [together_expr binds_statics]; try reflexivity.
    + apply together_expr_binds_statics.
    + rewrite together_expr_binds_statics. reflexivity.
    + rewrite together_handler_binds_statics, together_expr_binds_statics.
      reflexivity.
    + rewrite together_expr_binds_statics. f_equal.
      generalize (active_under sites handlers active). intros active'.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map existsb]. rewrite together_handler_binds_statics, IH.
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [together_handler handler_binds_statics].
    apply together_expr_binds_statics.
Qed.

Definition together_handlers (sites active : positions)
    (unboxed : list variable) (known : facts)
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map
    (fun '(k, h) =>
       (k,
        together_handler boxed sites active unboxed known
          (positions_of sites k) h))
    handlers.

Lemma together_handlers_bind_statics (sites active : positions)
    (unboxed : list variable) (known : facts)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics (together_handlers sites active unboxed known handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics, together_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map existsb]. rewrite together_handler_binds_statics, IH.
  reflexivity.
Qed.

Lemma lookup_together_handlers (sites active : positions)
    (unboxed : list variable) (known : facts) (k1 : continuation)
    (handlers : list (continuation * continuation_handler)) :
  lookup k1 (together_handlers sites active unboxed known handlers)
  = option_map
      (together_handler boxed sites active unboxed known
         (positions_of sites k1))
      (lookup k1 handlers).
Proof.
  unfold together_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map]. rewrite !lookup_cons, IH.
  destruct (Pos.eqb_spec k1 k') as [-> | Hne]; reflexivity.
Qed.

Lemma find_arm_together (active : positions) (unboxed : list variable)
    (known : facts) (j : Z) (arms : list (Z * apply_cont)) :
  find_arm j
    (map
       (fun '(n, ac) =>
          (n, together_apply_cont boxed active unboxed known ac))
       arms)
  = option_map (together_apply_cont boxed active unboxed known)
      (find_arm j arms).
Proof.
  induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
  cbn [map find_arm]. destruct (Z.eqb j n); [ reflexivity | exact IH ].
Qed.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

Section Runs.

Variable sites : positions.

Definition handlers_ok (active : positions) (unboxed : list variable)
    (known : facts) (handlers : list (continuation * continuation_handler))
    : bool :=
  forallb
    (fun '(k, h) =>
       together_handler_ok boxed sites active unboxed known
         (positions_of sites k) h)
    handlers.

(* What a continuation name stands for in the two runs, and which of
   the values that it is given are unboxed in the second: as in
   FlambdaUntagParamsProofs.v, with what is known where the closure
   was made, which holds of its variables at the heap and the record
   of the state that holds the continuations. *)
Inductive cont_rel (statics : static_env) (hp : heap) (rd : read_record)
    : list nat -> cont_value -> cont_value -> Prop :=
  | Rel_same (c : cont_value) : cont_rel statics hp rd [] c c
  | Rel_plain (hh : continuation_handler) (dv : env) (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    (∀ k1 : continuation,
      cont_rel statics hp rd [] (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel statics hp rd [] (Closure hh dv dc recursive)
      (Closure hh dv dc' recursive)
  | Rel_inside (us : list nat) (hh : continuation_handler) (dv dv' : env)
      (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler))
      (active : positions) (unboxed : list variable) (known : facts) :
    together_handler_ok boxed sites (active_under sites recursive active)
      unboxed known us hh
    = true ->
    handlers_ok (active_under sites recursive active) unboxed known recursive
    = true ->
    vars_rel statics unboxed dv dv' ->
    facts_hold known dv statics hp rd ->
    (∀ k1 : continuation,
      cont_rel statics hp rd (positions_of active k1) (resolve k1 dc)
        (resolve k1 dc')) ->
    cont_rel statics hp rd us (Closure hh dv dc recursive)
      (Closure
         (together_handler boxed sites (active_under sites recursive active)
            unboxed known us hh)
         dv' dc'
         (together_handlers sites (active_under sites recursive active)
            unboxed known recursive))
  | Rel_extra (c c' : cont_value) (extra : list value) :
    cont_rel statics hp rd [] c c' ->
    cont_rel statics hp rd [] (Extra_args c extra) (Extra_args c' extra)
  | Rel_over (remaining : list value) (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    cont_rel statics hp rd [] return_to return_to' ->
    cont_rel statics hp rd [] exn_to exn_to' ->
    cont_rel statics hp rd []
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Definition conts_rel (statics : static_env) (hp : heap) (rd : read_record)
    (active : positions) (conts conts' : cont_env) : Prop :=
  ∀ k1 : continuation,
    cont_rel statics hp rd (positions_of active k1) (resolve k1 conts)
      (resolve k1 conts').

Lemma cont_rel_extend (statics statics' : static_env) (hp : heap)
    (rd : read_record) (us : list nat) (c c' : cont_value) :
  cont_rel statics hp rd us c c' ->
  statics_extend statics statics' ->
  cont_rel statics' hp rd us c c'.
Proof.
  intros Hc Hmore.
  induction Hc
    as [c
       | hh dv dc dc' recursive Hdc IH
       | us hh dv dv' dc dc' recursive active unboxed known Hok Hrec Hvars
           Hfacts Hdc IH
       | c c' extra Hc IH
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise IHraise].
  - apply Rel_same.
  - apply Rel_plain. exact IH.
  - apply Rel_inside; try assumption.
    + exact (vars_rel_extend statics statics' unboxed dv dv' Hvars Hmore).
    + exact (facts_hold_extend known dv statics statics' hp rd Hfacts Hmore).
  - apply Rel_extra, IH.
  - apply Rel_over; assumption.
Qed.

Lemma conts_rel_extend (statics statics' : static_env) (hp : heap)
    (rd : read_record) (active : positions) (conts conts' : cont_env) :
  conts_rel statics hp rd active conts conts' ->
  statics_extend statics statics' ->
  conts_rel statics' hp rd active conts conts'.
Proof.
  intros Hconts Hmore k1.
  exact (cont_rel_extend statics statics' hp rd _ _ _ (Hconts k1) Hmore).
Qed.

(* A heap and a record that the ones of the relation evolve to. *)
Lemma cont_rel_evolves (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (us : list nat) (c c' : cont_value) :
  cont_rel statics hp rd us c c' ->
  memory_evolves hp rd hp' rd' ->
  cont_rel statics hp' rd' us c c'.
Proof.
  intros Hc Hevolves.
  induction Hc
    as [c
       | hh dv dc dc' recursive Hdc IH
       | us hh dv dv' dc dc' recursive active unboxed known Hok Hrec Hvars
           Hfacts Hdc IH
       | c c' extra Hc IH
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise IHraise].
  - apply Rel_same.
  - apply Rel_plain. exact IH.
  - apply Rel_inside; try assumption.
    exact (facts_hold_evolves known dv statics hp hp' rd rd' Hfacts Hevolves).
  - apply Rel_extra, IH.
  - apply Rel_over; assumption.
Qed.

Lemma conts_rel_evolves (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (active : positions) (conts conts' : cont_env) :
  conts_rel statics hp rd active conts conts' ->
  memory_evolves hp rd hp' rd' ->
  conts_rel statics hp' rd' active conts conts'.
Proof.
  intros Hconts Hevolves k1.
  exact (cont_rel_evolves statics hp hp' rd rd' _ _ _ (Hconts k1) Hevolves).
Qed.

(* Two states: the same expression, under the same variables; or an
   expression that the test accepts and the same rewritten, where what
   is known holds of the variables of the first. *)
Inductive state_rel : state -> state -> Prop :=
  | Rel_state_plain (e : expr) (vars : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) :
    conts_rel statics hp rd [] conts conts' ->
    state_rel (State e vars conts statics hp rd)
      (State e vars conts' statics hp rd)
  | Rel_state_inside (e : expr) (vars vars' : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) (active : positions)
      (unboxed : list variable) (known : facts) :
    together_ok boxed sites active unboxed known e = true ->
    vars_rel statics unboxed vars vars' ->
    facts_hold known vars statics hp rd ->
    conts_rel statics hp rd active conts conts' ->
    state_rel (State e vars conts statics hp rd)
      (State (together_expr boxed sites active unboxed known e) vars' conts'
         statics hp rd).

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

Lemma with_extra_args_rel (statics : static_env) (hp : heap)
    (rd : read_record) (c c' : cont_value) (extra : list value) :
  cont_rel statics hp rd [] c c' ->
  cont_rel statics hp rd [] (with_extra_args c extra)
    (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; simpl; [ exact Hc | apply Rel_extra, Hc ].
Qed.

(* The closures of the handlers of a recursive [Let_cont]. *)
Lemma rec_conts_plain (statics : static_env) (hp : heap) (rd : read_record)
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts conts' : cont_env) :
  conts_rel statics hp rd [] conts conts' ->
  conts_rel statics hp rd [] (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers vars conts' ++ conts').
Proof.
  intros Hconts k1. rewrite !resolve_rec_closures.
  destruct (lookup k1 handlers) as [hh | ]; [ | apply Hconts ].
  apply Rel_plain. exact Hconts.
Qed.

Lemma rec_conts_inside (statics : static_env) (hp : heap) (rd : read_record)
    (handlers : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) (active : positions)
    (unboxed : list variable) (known : facts) :
  handlers_ok (active_under sites handlers active) unboxed known handlers
  = true ->
  vars_rel statics unboxed vars vars' ->
  facts_hold known vars statics hp rd ->
  conts_rel statics hp rd active conts conts' ->
  conts_rel statics hp rd (active_under sites handlers active)
    (rec_closures handlers vars conts ++ conts)
    (rec_closures
       (together_handlers sites (active_under sites handlers active) unboxed
          known handlers)
       vars' conts'
     ++ conts').
Proof.
  intros Hok Hvars Hfacts Hconts k1.
  rewrite !resolve_rec_closures, lookup_together_handlers,
    Untag.positions_of_active_under.
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
  cont_rel statics hp rd [] return_to return_to' ->
  cont_rel statics hp rd [] exn_to exn_to' ->
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
  cont_rel statics hp rd [] return_to return_to' ->
  cont_rel statics hp rd [] exn_to exn_to' ->
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

(* Related continuations, applied to values and to the same with those
   at the unboxed positions unboxed. *)
Lemma apply_cont_value_rel (statics : static_env) (us : list nat)
    (c c' : cont_value) (hp : heap) (rd : read_record) :
  cont_rel statics hp rd us c c' ->
  ∀ vs ws : list value,
    Forall2 (unboxes_to statics) (moved_from us 0 vs) ws ->
    result_simulated state_rel (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' (kept_from us 0 vs ++ rev ws) statics hp rd).
Proof.
  intros Hc.
  induction Hc
    as [c
       | [ps m he] dv dc dc' recursive Hdc _
       | us [ps m he] dv dv' dc dc' recursive active unboxed known Hok Hrec
           Hvars Hfacts Hdc _
       | c c' extra Hc IHc
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise _];
    intros vs ws Hrel.
  - rewrite (plain_values vs ws Hrel).
    apply simulated_of_related, result_related_refl, state_rel_refl.
  - rewrite (plain_values vs ws Hrel). simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    destruct (bind_params ps vs dv) as [bound | ]; [ | exact I ].
    apply Rel_state_plain, rec_conts_plain. exact Hdc.
  - cbn [together_handler apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m))
      eqn:Hkinds; [ | exact I ].
    destruct (bind_params ps vs dv) as [bound | ] eqn:Hbound; [ | exact I ].
    cbn [together_handler_ok] in Hok.
    apply andb_true_iff in Hok as [Hfits Hok].
    unfold handler_fits in Hfits.
    apply andb_true_iff in Hfits as [Hdistinct _].
    rewrite (kinds_unboxed statics us vs ws m Hrel Hkinds).
    destruct
      (bind_params_rel statics unboxed us ps vs ws dv dv' bound Hdistinct
         Hvars Hrel Hbound)
      as (bound' & Hbound' & Hrel').
    rewrite Hbound'.
    apply Rel_state_inside; [ exact Hok | exact Hrel' | | ].
    + exact (facts_hold_bind_params statics hp rd ps known vs dv bound
        Hfacts Hbound).
    + apply rec_conts_inside; assumption.
  - rewrite (plain_values vs ws Hrel). cbn [apply_cont_value].
    specialize (IHc (vs ++ extra) []).
    rewrite Untag.kept_from_nil in IHc. cbn [rev] in IHc.
    rewrite app_nil_r in IHc.
    apply IHc. rewrite Untag.moved_from_nil. constructor.
  - rewrite (plain_values vs ws Hrel), !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; try exact I.
    apply run_call_rel; [ exact Hreturn | exact Hraise | ].
    intros partial. specialize (IHreturn [partial] []).
    rewrite Untag.kept_from_nil in IHreturn. cbn [rev] in IHreturn.
    rewrite app_nil_r in IHreturn.
    apply IHreturn. rewrite Untag.moved_from_nil. constructor.
Qed.

Lemma apply_cont_value_plain (statics : static_env) (c c' : cont_value)
    (vs : list value) (hp : heap) (rd : read_record) :
  cont_rel statics hp rd [] c c' ->
  result_simulated state_rel (apply_cont_value c vs statics hp rd)
    (apply_cont_value c' vs statics hp rd).
Proof.
  intros Hc.
  pose proof (apply_cont_value_rel statics [] c c' hp rd Hc vs [])
    as Happly.
  rewrite Untag.kept_from_nil in Happly. cbn [rev] in Happly.
  rewrite app_nil_r in Happly.
  apply Happly. rewrite Untag.moved_from_nil. constructor.
Qed.

Lemma call_function_rel (callee : value) (call_kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel statics hp rd [] return_to return_to' ->
  cont_rel statics hp rd [] exn_to exn_to' ->
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

(* Two related states take a step each, to related states or to the
   same exit, unless the first is undefined. *)
Lemma together_step (s s' : state) :
  state_rel s s' -> result_simulated state_rel (step s) (step s').
Proof using Type.
  intros
    [e vars conts conts' statics hp rd Hconts
    | e vars vars' conts conts' statics hp rd active unboxed known Hok Hvars
        Hfacts Hconts].
  - (* the same expression *)
    destruct e as [y kind1 d body | bindings body | k1 hh body
                  | handlers body | ap | ac | scrutinee arms | ].
    + rewrite !step_let_exec.
      destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ] eqn:Hv;
        [ | exact I ].
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      apply Rel_state_plain.
      exact (conts_rel_evolves statics hp hp' rd rd' [] conts conts' Hconts
        (exec_named_memory_evolves statics hp hp' rd rd' vars d v Hv)).
    + rewrite !step_let_static.
      destruct (eval_static_bindings vars bindings statics) as [extended | ]
        eqn:Hextended; [ | exact I ].
      apply Rel_state_plain.
      exact (conts_rel_extend statics extended hp rd [] conts conts' Hconts
        (eval_static_bindings_extend vars bindings _ _ Hextended)).
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
      cbn [together_expr together_ok] in Hok |- *.
    + (* Let: the unboxing of an unboxed variable reads no cell, and is
         the variable in the second run; any other defining expression
         names no unboxed variable, and runs alike in the two. *)
      apply andb_true_iff in Hok as [Hd Hbody].
      rewrite !step_let_exec.
      destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ]
        eqn:Hv; [ | exact I ].
      assert (Hv' :
        exec_named statics hp rd vars' (together_named boxed unboxed d)
        = Some (v, hp', rd')).
      { unfold together_named_ok in Hd. unfold together_named.
        destruct (reads_unboxed boxed unboxed d) eqn:Hreads.
        - unfold reads_unboxed in Hreads.
          destruct (unbox_of boxed d) as [p | ] eqn:Hunbox;
            [ | discriminate ].
          rewrite Hreads. apply unbox_of_some in Hunbox. subst d.
          destruct
            (exec_named_pure statics hp hp' rd rd' vars
               (Prim (Unary (Unbox_number boxed)) [Var p]) v eq_refl Hv)
            as (Heval & -> & ->).
          rewrite exec_named_not_prim by (intros ? ?; discriminate).
          rewrite (eval_unboxing statics unboxed vars vars' p v Hvars Hreads
            Heval).
          reflexivity.
        - cbn [orb] in Hd.
          assert (Hsame :
            exec_named statics hp rd vars' d
            = exec_named statics hp rd vars d)
            by exact
                 (exec_named_plain statics statics hp rd unboxed vars vars' d
                    Hvars Hd).
          unfold reads_unboxed in Hreads.
          destruct (unbox_of boxed d) as [p | ]; [ rewrite Hreads | ];
            rewrite Hsame; exact Hv. }
      rewrite Hv'.
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      apply Rel_state_inside;
        [ exact Hbody | apply vars_rel_cons, Hvars | | ].
      * exact (facts_hold_under_let known y d vars statics hp hp' rd rd' v
          Hfacts Hv).
      * exact (conts_rel_evolves statics hp hp' rd rd' active conts conts'
          Hconts
          (exec_named_memory_evolves statics hp hp' rd rd' vars d v Hv)).
    + apply andb_true_iff in Hok as [Hbindings Hbody].
      rewrite !step_let_static,
        (eval_static_bindings_plain statics unboxed vars vars' bindings
           statics
           Hvars Hbindings).
      destruct (eval_static_bindings vars bindings statics) as [extended | ]
        eqn:Hextended; [ | exact I ].
      pose proof (eval_static_bindings_extend vars bindings _ _ Hextended)
        as Hmore.
      apply Rel_state_inside; [ exact Hbody | | | ].
      * exact (vars_rel_extend statics extended unboxed vars vars' Hvars
          Hmore).
      * exact (facts_hold_under_statics known vars statics bindings extended
          hp rd Hfacts Hextended).
      * exact (conts_rel_extend statics extended hp rd active conts conts'
          Hconts Hmore).
    + apply andb_true_iff in Hok as [Hhh Hbody].
      rewrite !step_let_cont_non_recursive.
      apply Rel_state_inside; [ exact Hbody | exact Hvars | exact Hfacts | ].
      intros k2. rewrite Untag.positions_of_cons.
      destruct (Pos.eqb_spec k2 k1) as [-> | Hne].
      * rewrite !resolve_head.
        exact
          (Rel_inside statics hp rd (positions_of sites k1) hh vars vars'
             conts conts' [] active unboxed known Hhh eq_refl Hvars Hfacts
             Hconts).
      * rewrite !(resolve_other k2 k1) by exact Hne. exact (Hconts k2).
    + apply andb_true_iff in Hok as [Hhandlers Hbody].
      rewrite !step_let_cont_recursive.
      fold
        (together_handlers sites (active_under sites handlers active) unboxed
           known handlers).
      rewrite together_handlers_bind_statics.
      destruct (handlers_bind_statics handlers); [ exact I | ].
      apply Rel_state_inside; [ exact Hbody | exact Hvars | exact Hfacts | ].
      apply rec_conts_inside; assumption.
    + apply andb_true_iff in Hok as [Hok Hraise].
      apply andb_true_iff in Hok as [Hargs Hreturn].
      assert (Hcallee :
        none_in unboxed (fun p => var_in_simple p (ap_callee ap)) = true).
      { unfold none_in in Hargs |- *. rewrite forallb_forall in Hargs.
        apply forallb_forall. intros p Hp. specialize (Hargs p Hp).
        apply negb_true_iff in Hargs. apply orb_false_iff in Hargs as
          [Hargs _].
        apply orb_false_iff in Hargs as [Hargs _]. rewrite Hargs.
        reflexivity. }
      assert (Hap_args :
        none_in unboxed (fun p => var_in_simples p (ap_args ap)) = true).
      { unfold none_in in Hargs |- *. rewrite forallb_forall in Hargs.
        apply forallb_forall. intros p Hp. specialize (Hargs p Hp).
        apply negb_true_iff in Hargs. apply orb_false_iff in Hargs as
          [Hargs _].
        apply orb_false_iff in Hargs as [_ Hargs]. rewrite Hargs.
        reflexivity. }
      assert (Hextra :
        none_in unboxed (fun p => var_in_simples p (exn_extra_simples ap))
        = true).
      { unfold none_in in Hargs |- *. rewrite forallb_forall in Hargs.
        apply forallb_forall. intros p Hp. specialize (Hargs p Hp).
        apply negb_true_iff in Hargs. apply orb_false_iff in Hargs as
          [_ Hargs].
        rewrite Hargs. reflexivity. }
      rewrite !step_apply,
        (eval_simple_plain statics unboxed vars vars' _ Hvars Hcallee),
        (eval_simples_plain statics unboxed vars vars' _ Hvars Hap_args),
        (eval_simples_plain statics unboxed vars vars' _ Hvars Hextra).
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
    + rewrite !step_apply_cont. unfold together_apply_cont.
      cbn [ac_args ac_continuation].
      destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
        [ | exact I ].
      unfold together_apply_cont_ok in Hok.
      destruct
        (eval_together_args statics hp rd unboxed known _ vars vars'
           (ac_args ac) vs Hvars Hfacts Hok Hvs)
        as (ws & Hvs' & Hrel).
      rewrite Hvs'. unfold apply_continuation.
      exact
        (apply_cont_value_rel statics _ _ _ hp rd
           (Hconts (ac_continuation ac)) vs ws Hrel).
    + apply andb_true_iff in Hok as [Hscrutinee Harms].
      rewrite !step_switch,
        (eval_simple_plain statics unboxed vars vars' _ Hvars Hscrutinee).
      destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
      destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
      rewrite find_arm_together.
      destruct (find_arm j arms) as [ac | ] eqn:Hfind; [ | exact I ].
      cbn [option_map].
      rewrite forallb_forall in Harms.
      exact
        (Rel_state_inside (Apply_cont ac) vars vars' conts conts' statics hp
           rd active unboxed known
           (Harms (j, ac) (find_arm_in j arms ac Hfind)) Hvars Hfacts
           Hconts).
    + rewrite !step_invalid. exact I.
Qed.

Lemma related_refines (s s' : state) : state_rel s s' -> refines s s'.
Proof. exact (simulation_refines state_rel together_step s s'). Qed.

End Runs.

(* The rewrite at the positions given refines, where what is known
   holds. *)
Lemma together_at_refines (sites : positions) (known : facts) (e : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  facts_hold known vars statics hp rd ->
  refines (State e vars conts statics hp rd)
    (State (together_at boxed sites known e) vars conts statics hp rd).
Proof.
  intros Hfacts. unfold together_at.
  destruct (together_ok boxed sites [] [] known e) eqn:Hok;
    [ | apply refines_refl ].
  apply (related_refines sites).
  apply Rel_state_inside;
    [ exact Hok | apply vars_rel_nil | exact Hfacts | ].
  intros k1. apply Rel_same.
Qed.

End Sort.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Lemma unbox_together_all_refines (boxes : list boxable_number) :
  ∀ (known : facts) (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    facts_hold known vars statics hp rd ->
    refines (State e vars conts statics hp rd)
      (State (unbox_together_all boxes known e) vars conts statics hp rd).
Proof.
  induction boxes as [ | boxed boxes IH];
    intros known e vars conts statics hp rd Hfacts; cbn [unbox_together_all];
    [ apply refines_refl | ].
  eapply refines_trans; [ | apply IH; exact Hfacts ].
  unfold unbox_together_of. apply together_at_refines, Hfacts.
Qed.

Theorem unbox_together_knowing_symbols_refines :
  unbox_together_knowing_symbols_refines_statement.
Proof.
  intros known e vars conts statics hp rd Honly Hbound.
  unfold unbox_together_knowing. apply unbox_together_all_refines.
  exact (symbols_facts_hold known vars statics hp rd Honly Hbound).
Qed.
