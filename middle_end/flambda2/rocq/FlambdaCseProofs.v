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

(* Proofs of the statements of FlambdaCseSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextProofs FlambdaContextMemoryProofs
  FlambdaBindersSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaApplyProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaHeapSpec
  FlambdaHeapProofs FlambdaCseSpec FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* What may be shared stays                                                 *)
(* ======================================================================== *)

Theorem shareable_stable : shareable_stable_statement.
Proof.
  intros p Hshareable.
  destruct p as [u | b | v | hp | rp]; try discriminate Hshareable.
  - destruct u; try discriminate Hshareable; try reflexivity.
    all: match goal with
         | m : mutability |- _ => destruct m
         end; first [ discriminate Hshareable | reflexivity ].
  - destruct b; try discriminate Hshareable; try reflexivity.
    all: match goal with
         | m : mutability |- _ => destruct m
         end; first [ discriminate Hshareable | reflexivity ].
Qed.

(* ======================================================================== *)
(* What is known stays so                                                   *)
(* ======================================================================== *)

(* Binding a variable, once what mentions it is forgotten: the run of a
   primitive reads the variables through the values of its arguments
   only (FlambdaSemanticsSpec.exec_named_args_statement). *)
Lemma all_computed_forget (known : computed) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) (x : variable)
    (v : value) :
  all_computed known vars statics hp rd ->
  all_computed (forget_computed x known) ((x, v) :: vars) statics hp rd.
Proof.
  intros Hhold entry Hin. unfold forget_computed in Hin.
  apply filter_In in Hin. destruct Hin as [Hin Hkept]. cbv beta in Hkept.
  apply andb_true_iff in Hkept as [Hkey Hargs].
  apply negb_true_iff, Pos.eqb_neq in Hkey. apply negb_true_iff in Hargs.
  destruct (Hhold entry Hin) as (Hstable & r & Hexec & Hlookup).
  split; [ exact Hstable | ]. exists r. split.
  - rewrite <- Hexec. apply exec_named_args, eval_simples_extend.
    intros Hfree. apply var_in_simples_sound in Hfree.
    exact (eq_true_false_abs _ Hfree Hargs).
  - simpl. destruct (Pos.eqb_spec (fst entry) x) as [Heq | _];
      [ | exact Hlookup ].
    exfalso. apply Hkey. symmetry. exact Heq.
Qed.

Lemma all_computed_bind_params (statics : static_env) (hp : heap)
    (rd : read_record) (params : list variable) :
  ∀ (known : computed) (vs : list value) (vars handler_vars : env),
    all_computed known vars statics hp rd ->
    bind_params params vs vars = Some handler_vars ->
    all_computed (forget_computed_vars params known) handler_vars statics hp
      rd.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, all_computed_forget, Hhold.
Qed.

(* A static environment that binds more: the run of a primitive is the
   same run (FlambdaSemanticsSpec.exec_named_statics_extend_statement). *)
Lemma all_computed_extend (known : computed) (vars : env)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  all_computed known vars statics hp rd -> statics_extend statics statics' ->
  all_computed known vars statics' hp rd.
Proof.
  intros Hhold [_ Hvalues] entry Hin.
  destruct (Hhold entry Hin) as (Hstable & v & Hexec & Hlookup).
  split; [ exact Hstable | ]. exists v. split; [ | exact Hlookup ].
  apply (exec_named_statics_extend statics statics' hp hp rd rd vars _ _ v);
    [ | exact Hexec ].
  intros vs s allocated _ _ Hs. apply Hvalues, Hs.
Qed.

(* A heap and a record that those have evolved to: a primitive whose
   value stays runs there to the same
   (FlambdaSemanticsSpec.stable_prim_stays_statement). *)
Lemma all_computed_evolves (known : computed) (vars : env)
    (statics : static_env) (hp hp' : heap) (rd rd' : read_record) :
  all_computed known vars statics hp rd -> memory_evolves hp rd hp' rd' ->
  all_computed known vars statics hp' rd'.
Proof.
  intros Hhold Hevolves entry Hin.
  destruct (Hhold entry Hin) as (Hstable & v & Hexec & Hlookup).
  split; [ exact Hstable | ]. exists v. split; [ | exact Hlookup ].
  exact
    (stable_prim_stays statics hp hp' rd rd' vars _ _ v Hstable Hexec
       Hevolves).
Qed.

(* Where an untagging has a value, the tagging of that value is what
   was untagged. *)
Lemma tag_of_untagged (statics : static_env) (tagged v : value) :
  eval_prim statics (Unary Untag_immediate) [tagged] = Some v ->
  eval_prim statics (Unary Tag_immediate) [v] = Some tagged.
Proof.
  rewrite !eval_prim_unary. unfold eval_unary_primitive, int_of_value.
  destruct tagged; try discriminate.
  match goal with c : const |- _ => destruct c; try discriminate end.
  simpl.
  match goal with
  | |- context [in_range ?w ?i] =>
    destruct (in_range w i) eqn:Hrange; [ | discriminate ]
  end.
  intros [= <-]. simpl. rewrite Hrange. reflexivity.
Qed.

(* And that value is a number, not a location. *)
Lemma untagged_no_location (statics : static_env) (tagged v : value) :
  eval_prim statics (Unary Untag_immediate) [tagged] = Some v ->
  ∀ l : location, Value_location l ∉ [v].
Proof.
  rewrite eval_prim_unary. unfold eval_unary_primitive, int_of_value.
  destruct tagged; try discriminate.
  match goal with c : const |- _ => destruct c; try discriminate end.
  simpl.
  match goal with
  | |- context [in_range ?w ?i] =>
    destruct (in_range w i); [ | discriminate ]
  end.
  intros [= <-] l [Heq | []]. discriminate Heq.
Qed.

(* What [tagged_again] adds is of the one form. *)
Lemma tagged_again_spec (x : variable) (p : prim) (args : list simple)
    (entry : variable * (prim * list simple)) :
  In entry (tagged_again x p args) ->
  ∃ t : variable,
    p = Unary Untag_immediate
    ∧ args = [Var t]
    ∧ entry = (t, (Unary Tag_immediate, [Var x])).
Proof.
  unfold tagged_again.
  destruct p as [u | | | | ]; try contradiction.
  destruct u; try contradiction.
  destruct args as [ | [t | s | c] [ | arg args]]; try contradiction.
  intros [<- | []]. exists t. auto.
Qed.

(* The variable that it says something of is one of the arguments. *)
Lemma tagged_again_free (x : variable) (p : prim) (args : list simple)
    (entry : variable * (prim * list simple)) :
  In entry (tagged_again x p args) ->
  In (fst entry) (free_variables_of_named (Prim p args)).
Proof.
  intros Hin.
  destruct (tagged_again_spec x p args entry Hin) as (t & -> & -> & ->).
  left. reflexivity.
Qed.

(* Under a [Let], at the heap and the record that it leaves. What was
   known holds there, as they have evolved to them. The primitive of
   the [Let], whose value stays, runs again there to the same
   (FlambdaSemanticsSpec.stable_prim_again_statement). An untagging
   reads no cell, so it left the heap and the record as they were, and
   the tagging of its value runs by its value. *)
Lemma all_computed_under_let (known : computed) (x : variable) (d : named)
    (vars : env) (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (v : value) :
  all_computed known vars statics hp rd ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  all_computed (computed_under_let known x d) ((x, v) :: vars) statics hp'
    rd'.
Proof.
  intros Hhold Hd. unfold computed_under_let.
  pose proof
    (all_computed_forget known vars statics hp' rd' x v
       (all_computed_evolves known vars statics hp hp' rd rd' Hhold
          (exec_named_memory_evolves statics hp hp' rd rd' vars d v Hd)))
    as Hforgot.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hforgot.
  destruct (shareable p && negb (var_in_simples x args)) eqn:Hadd;
    [ | exact Hforgot ].
  apply andb_true_iff in Hadd as [Hshareable Hmention].
  apply negb_true_iff in Hmention.
  pose proof (shareable_stable p Hshareable) as Hstable.
  intros entry [<- | Hin].
  - split; [ exact Hstable | ]. exists v. cbn [fst snd]. split.
    + rewrite
        (exec_named_args statics hp' rd' vars ((x, v) :: vars) p args args).
      * exact
          (stable_prim_again statics hp hp' rd rd' vars p args v Hstable Hd).
      * apply eval_simples_extend. intros Hfree.
        apply var_in_simples_sound in Hfree. congruence.
    + simpl. rewrite Pos.eqb_refl. reflexivity.
  - apply in_app_or in Hin as [Hagain | Hin]; [ | exact (Hforgot entry Hin) ].
    destruct (tagged_again_spec x p args entry Hagain) as (t & -> & -> & ->).
    assert (Hxt : Pos.eqb x t = false).
    { cbn [var_in_simples var_in_simple existsb] in Hmention.
      rewrite orb_false_r in Hmention. exact Hmention. }
    destruct
      (exec_named_eval_named statics hp hp' rd rd' vars
         (Unary Untag_immediate) [Var t] v eq_refl Hd)
      as (Heval & -> & ->).
    rewrite eval_named_prim in Heval.
    cbn [eval_simples eval_simple] in Heval.
    destruct (lookup t vars) as [tagged | ] eqn:Ht; [ | discriminate Heval ].
    assert
      (Huntag : eval_prim statics (Unary Untag_immediate) [tagged] = Some v)
      by exact Heval.
    split; [ reflexivity | ]. exists tagged. cbn [fst snd]. split.
    + apply
        (eval_named_exec_named statics hp rd ((x, v) :: vars)
           (Unary Tag_immediate) [Var x] [v] tagged).
      * reflexivity.
      * cbn [eval_simples eval_simple lookup]. rewrite Pos.eqb_refl.
        reflexivity.
      * intros l Hl. exfalso.
        exact (untagged_no_location statics tagged v Huntag l Hl).
      * rewrite eval_named_prim.
        cbn [eval_simples eval_simple lookup]. rewrite Pos.eqb_refl.
        exact (tag_of_untagged statics tagged v Huntag).
    + cbn [lookup]. rewrite Pos.eqb_sym, Hxt. exact Ht.
Qed.

(* ======================================================================== *)
(* What is known of the arrays made stays so                                *)
(* ======================================================================== *)

(* The lemmas above, of one entry of a primitive that may be shared. *)
Lemma computed_holds_forget (statics : static_env) (hp : heap)
    (rd : read_record) (vars : env) (x : variable) (v : value)
    (entry : variable * (prim * list simple)) :
  Pos.eqb x (fst entry) = false ->
  var_in_simples x (snd (snd entry)) = false ->
  computed_holds statics hp rd vars entry ->
  computed_holds statics hp rd ((x, v) :: vars) entry.
Proof.
  intros Hkey Hargs (Hstable & r & Hexec & Hlookup).
  split; [ exact Hstable | ]. exists r. split.
  - rewrite <- Hexec. apply exec_named_args, eval_simples_extend.
    intros Hfree. apply var_in_simples_sound in Hfree.
    exact (eq_true_false_abs _ Hfree Hargs).
  - simpl. rewrite Pos.eqb_sym, Hkey. exact Hlookup.
Qed.

Lemma computed_holds_extend (statics statics' : static_env) (hp : heap)
    (rd : read_record) (vars : env)
    (entry : variable * (prim * list simple)) :
  statics_extend statics statics' ->
  computed_holds statics hp rd vars entry ->
  computed_holds statics' hp rd vars entry.
Proof.
  intros [_ Hvalues] (Hstable & v & Hexec & Hlookup).
  split; [ exact Hstable | ]. exists v. split; [ | exact Hlookup ].
  apply (exec_named_statics_extend statics statics' hp hp rd rd vars _ _ v);
    [ | exact Hexec ].
  intros vs s allocated _ _ Hs. apply Hvalues, Hs.
Qed.

Lemma computed_holds_evolves (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env)
    (entry : variable * (prim * list simple)) :
  memory_evolves hp rd hp' rd' ->
  computed_holds statics hp rd vars entry ->
  computed_holds statics hp' rd' vars entry.
Proof.
  intros Hevolves (Hstable & v & Hexec & Hlookup).
  split; [ exact Hstable | ]. exists v. split; [ | exact Hlookup ].
  exact
    (stable_prim_stays statics hp hp' rd rd' vars _ _ v Hstable Hexec
       Hevolves).
Qed.

(* The same of one entry of the allocation of a mutable array. Binding
   a variable that is not that of the entry. *)
Lemma array_made_holds_forget (statics : static_env) (hp : heap)
    (vars : env) (x : variable) (v : value)
    (entry : variable * (prim * list simple)) :
  Pos.eqb x (fst entry) = false ->
  array_made_holds statics hp vars entry ->
  array_made_holds statics hp ((x, v) :: vars) entry.
Proof.
  intros Hkey
    (kind & h & h1 & vs & w & Hprim & Hlength & Hexec & Hevolved & Hlookup).
  exists kind, h, h1, vs, w.
  split; [ exact Hprim | ]. split; [ exact Hlength | ].
  split; [ exact Hexec | ]. split; [ exact Hevolved | ].
  simpl. rewrite Pos.eqb_sym, Hkey. exact Hlookup.
Qed.

(* A static environment that binds more: the allocation runs the same
   (FlambdaHeapSpec.exec_prim_statics_extend_statement). *)
Lemma array_made_holds_extend (statics statics' : static_env) (hp : heap)
    (vars : env) (entry : variable * (prim * list simple)) :
  statics_extend statics statics' ->
  array_made_holds statics hp vars entry ->
  array_made_holds statics' hp vars entry.
Proof.
  intros [_ Hvalues]
    (kind & h & h1 & vs & w & Hprim & Hlength & Hexec & Hevolved & Hlookup).
  exists kind, h, h1, vs, w.
  split; [ exact Hprim | ]. split; [ exact Hlength | ].
  split; [ | split; [ exact Hevolved | exact Hlookup ] ].
  apply (exec_prim_statics_extend statics statics' h h1 _ vs w);
    [ | exact Hexec ].
  intros s allocated _ Hs. apply Hvalues, Hs.
Qed.

(* A heap that the heap has evolved to. *)
Lemma array_made_holds_evolves (statics : static_env) (hp hp' : heap)
    (vars : env) (entry : variable * (prim * list simple)) :
  heap_evolves hp hp' ->
  array_made_holds statics hp vars entry ->
  array_made_holds statics hp' vars entry.
Proof.
  intros Hevolves
    (kind & h & h1 & vs & w & Hprim & Hlength & Hexec & Hevolved & Hlookup).
  exists kind, h, h1, vs, w.
  split; [ exact Hprim | ]. split; [ exact Hlength | ].
  split; [ exact Hexec | ].
  split; [ exact (heap_evolves_trans h1 hp hp' Hevolved Hevolves) | ].
  exact Hlookup.
Qed.

(* Under the [Let] of the allocation, at the heap that it leaves. *)
Lemma array_made_new (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (x : variable) (d : named)
    (v : value) (entry : variable * (prim * list simple)) :
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  In entry (array_made_under_let x d) ->
  array_made_holds statics hp' ((x, v) :: vars) entry.
Proof.
  intros Hd Hin. unfold array_made_under_let in Hin.
  destruct d as [s | p args | | | | | ]; try contradiction.
  destruct p as [ | | | hprim | ]; try contradiction.
  destruct hprim as [ | | kind | ]; try contradiction.
  destruct (var_in_simples x args); [ contradiction | ].
  destruct Hin as [<- | []].
  cbn [exec_named] in Hd.
  destruct (eval_simples vars args) as [vs | ] eqn:Hvs; [ | discriminate Hd ].
  cbv beta iota in Hd.
  destruct (exec_prim statics hp (Heap (Make_mutable_array kind)) vs)
    as [[w h1] | ] eqn:Hexec; [ | discriminate Hd ].
  injection Hd as <- <- <-.
  exists kind, hp, h1, vs, w. cbn [fst snd].
  split; [ reflexivity | ].
  split; [ exact (eval_simples_length vars args vs Hvs) | ].
  split; [ exact Hexec | ]. split; [ apply heap_evolves_refl | ].
  simpl. rewrite Pos.eqb_refl. reflexivity.
Qed.

(* What the sharing knows, of the two sorts of entries. *)
Lemma all_computed_or_made_forget (known : computed) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) (x : variable)
    (v : value) :
  all_computed_or_made known vars statics hp rd ->
  all_computed_or_made (forget_computed x known) ((x, v) :: vars) statics hp
    rd.
Proof.
  intros Hhold entry Hin. unfold forget_computed in Hin.
  apply filter_In in Hin. destruct Hin as [Hin Hkept]. cbv beta in Hkept.
  apply andb_true_iff in Hkept as [Hkey Hargs].
  apply negb_true_iff in Hkey. apply negb_true_iff in Hargs.
  destruct (Hhold entry Hin) as [Hcomputed | Hmade].
  - left.
    exact
      (computed_holds_forget statics hp rd vars x v entry Hkey Hargs
         Hcomputed).
  - right.
    exact (array_made_holds_forget statics hp vars x v entry Hkey Hmade).
Qed.

Lemma all_computed_or_made_bind_params (statics : static_env) (hp : heap)
    (rd : read_record) (params : list variable) :
  ∀ (known : computed) (vs : list value) (vars handler_vars : env),
    all_computed_or_made known vars statics hp rd ->
    bind_params params vs vars = Some handler_vars ->
    all_computed_or_made (forget_computed_vars params known) handler_vars
      statics hp rd.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, all_computed_or_made_forget, Hhold.
Qed.

Lemma all_computed_or_made_extend (known : computed) (vars : env)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  all_computed_or_made known vars statics hp rd ->
  statics_extend statics statics' ->
  all_computed_or_made known vars statics' hp rd.
Proof.
  intros Hhold Hextend entry Hin.
  destruct (Hhold entry Hin) as [Hcomputed | Hmade].
  - left.
    exact
      (computed_holds_extend statics statics' hp rd vars entry Hextend
         Hcomputed).
  - right.
    exact
      (array_made_holds_extend statics statics' hp vars entry Hextend Hmade).
Qed.

Lemma all_computed_or_made_evolves (known : computed) (vars : env)
    (statics : static_env) (hp hp' : heap) (rd rd' : read_record) :
  all_computed_or_made known vars statics hp rd ->
  memory_evolves hp rd hp' rd' ->
  all_computed_or_made known vars statics hp' rd'.
Proof.
  intros Hhold Hevolves entry Hin.
  destruct (Hhold entry Hin) as [Hcomputed | Hmade].
  - left.
    exact
      (computed_holds_evolves statics hp hp' rd rd' vars entry Hevolves
         Hcomputed).
  - right.
    exact
      (array_made_holds_evolves statics hp hp' vars entry (proj1 Hevolves)
         Hmade).
Qed.

(* What [computed_under_let] gives is what it adds, which it adds to
   nothing known as well, or what it keeps. *)
Lemma computed_under_let_split (known : computed) (x : variable) (d : named)
    (entry : variable * (prim * list simple)) :
  In entry (computed_under_let known x d) ->
  In entry (computed_under_let [] x d) ∨ In entry (forget_computed x known).
Proof.
  unfold computed_under_let. cbv zeta. cbn [forget_computed filter].
  destruct d as [s | p args | | | | | ]; try (intros H; right; exact H).
  destruct (shareable p && negb (var_in_simples x args));
    [ | intros H; right; exact H ].
  intros [<- | Hin]; [ left; left; reflexivity | ].
  apply in_app_or in Hin as [Hagain | Hin]; [ | right; exact Hin ].
  left. right. apply in_or_app. left. exact Hagain.
Qed.

(* Under a [Let], at the heap and the record that it leaves: what
   [computed_under_let] adds holds as above, the array that the [Let]
   makes is known by an entry of its allocation, and what was known
   holds there still. *)
Lemma all_computed_or_made_under_let (known : computed) (x : variable)
    (d : named) (vars : env) (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (v : value) :
  all_computed_or_made known vars statics hp rd ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  all_computed_or_made (computed_or_made_under_let known x d)
    ((x, v) :: vars) statics hp' rd'.
Proof.
  intros Hhold Hd entry Hin. unfold computed_or_made_under_let in Hin.
  apply in_app_or in Hin as [Hmade | Hin].
  - right.
    exact (array_made_new statics hp hp' rd rd' vars x d v entry Hd Hmade).
  - apply computed_under_let_split in Hin as [Hnew | Hold].
    + left.
      apply
        (all_computed_under_let [] x d vars statics hp hp' rd rd' v);
        [ intros e [] | exact Hd | exact Hnew ].
    + exact
        (all_computed_or_made_forget known vars statics hp' rd' x v
           (all_computed_or_made_evolves known vars statics hp hp' rd rd'
              Hhold
              (exec_named_memory_evolves statics hp hp' rd rd' vars d v Hd))
           entry Hold).
Qed.

(* ======================================================================== *)
(* Sharing                                                                  *)
(* ======================================================================== *)

Lemma computed_by_sound (known : computed) (p : prim) (args : list simple)
    (x : variable) :
  computed_by known p args = Some x -> (x, (p, args)) ∈ known.
Proof.
  induction known as [ | [y [p' args']] known IH]; simpl; [ discriminate | ].
  destruct (prim_eqb p p' && same_simples args args') eqn:Hsame.
  - intros [= ->]. apply andb_true_iff in Hsame as [Hp Hargs].
    apply prim_eqb_sound in Hp. unfold same_simples in Hargs.
    destruct (list_eq_dec simple_eq_dec args args') as [Heq | ];
      [ | discriminate ].
    subst. left. reflexivity.
  - intros Hcomputed. right. apply IH, Hcomputed.
Qed.

(* An array that is known to have been made is so by an entry of the
   allocation of a mutable array, of as many arguments as its length
   says. *)
Lemma made_length_sound (known : computed) (x : variable) (n : Z) :
  made_length known x = Some n ->
  ∃ (kind : array_kind) (args : list simple),
    In (x, (Heap (Make_mutable_array kind), args)) known
    ∧ n = Z.of_nat (length args).
Proof.
  induction known as [ | [y [p args]] known IH]; cbn [made_length];
    [ discriminate | ].
  assert
    (Hrest :
      made_length known x = Some n ->
      ∃ (kind : array_kind) (args0 : list simple),
        In (x, (Heap (Make_mutable_array kind), args0))
          ((y, (p, args)) :: known)
        ∧ n = Z.of_nat (length args0)).
  { intros H. destruct (IH H) as (kind & args0 & Hin & Hn).
    exists kind, args0. split; [ right; exact Hin | exact Hn ]. }
  destruct p as [ | | | hprim | ]; try exact Hrest.
  destruct hprim as [ | | kind | ]; try exact Hrest.
  destruct (Pos.eqb_spec x y) as [-> | _]; [ | exact Hrest ].
  intros [= <-]. exists kind, args. split; [ left | ]; reflexivity.
Qed.

(* Where what is known holds, the constant that takes the place of the
   length of an array made runs as the length does: the length has a
   value on what the allocation gave, in every heap that the one it
   left evolves to (FlambdaHeapSpec.array_length_known_statement), and
   that value is the number of the arguments, which is a tagged
   immediate where the rule answers. *)
Lemma made_length_named_same (known : computed) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) (d d' : named) :
  all_computed_or_made known vars statics hp rd ->
  made_length_named known d = Some d' ->
  exec_named statics hp rd vars d' = exec_named statics hp rd vars d.
Proof.
  intros Hhold Hd'. unfold made_length_named in Hd'.
  destruct d as [s | p args | | | | | ]; try discriminate Hd'.
  destruct p as [u | | | | ]; try discriminate Hd'.
  destruct u; try discriminate Hd'.
  destruct args as [ | [x | sym | c] [ | arg args]]; try discriminate Hd'.
  destruct (made_length known x) as [n | ] eqn:Hmade; [ | discriminate Hd' ].
  destruct (in_range (int_width Standard_tagged_immediate) n) eqn:Hrange;
    [ | discriminate Hd' ].
  injection Hd' as <-.
  destruct (made_length_sound known x n Hmade) as (mkind & margs & Hin & ->).
  destruct (Hhold _ Hin) as [(Hstable & _) | Hholds];
    [ discriminate Hstable | ].
  destruct Hholds
    as (kind' & h & h1 & vs & v & _ & Hlength & Hexec & Hevolved & Hlookup).
  cbn [fst snd] in Hlength, Hlookup.
  cbn [exec_named eval_named eval_simples eval_simple]. rewrite Hlookup.
  cbv beta iota.
  rewrite (array_length_known statics h h1 hp kind' _ vs v Hexec Hevolved).
  unfold array_length_of. rewrite Hlength, Hrange. reflexivity.
Qed.

(* Where what is known holds, sharing does not change how a defining
   expression runs: the variable that takes the place of a primitive
   holds the value that the primitive gives there, and the primitive
   leaves the heap and the record as they are, as a variable does. An
   entry of the allocation of a mutable array is never what a lookup
   answers: the lookup is of a primitive that may be shared. *)
Lemma share_named_same (known : computed) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) (d : named) :
  all_computed_or_made known vars statics hp rd ->
  exec_named statics hp rd vars (share_named known d)
  = exec_named statics hp rd vars d.
Proof.
  intros Hhold. unfold share_named.
  destruct (made_length_named known d) as [d' | ] eqn:Hlength;
    [ exact (made_length_named_same known vars statics hp rd d d' Hhold Hlength)
    | ].
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try reflexivity.
  destruct (shareable p) eqn:Hshareable; [ | reflexivity ].
  destruct (computed_by known p args) as [x | ] eqn:Hcomputed;
    [ | reflexivity ].
  destruct (Hhold _ (computed_by_sound _ _ _ _ Hcomputed))
    as [(_ & v & Hrun & Hlookup) | Hmade].
  - cbn [fst snd] in Hrun, Hlookup. rewrite Hrun.
    cbn [exec_named eval_named eval_simple]. rewrite Hlookup. reflexivity.
  - destruct Hmade as (kind & _ & _ & _ & _ & Hprim & _).
    cbn [fst snd] in Hprim. subst p. discriminate Hshareable.
Qed.

(* What the pass puts for a named expression that runs runs to the same
   value, and leaves the same heap and the same record. *)
Lemma share_named_sound (known : computed) (vars : env)
    (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
    (d : named) (v : value) :
  all_computed_or_made known vars statics hp rd ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named statics hp rd vars (share_named known d) = Some (v, hp', rd').
Proof.
  intros Hhold Hd.
  rewrite (share_named_same known vars statics hp rd d Hhold). exact Hd.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* As in FlambdaUnboxProofs.v: at a [Let], the variable runs to what
   the primitive ran to or the primitive has no run; then the parts
   under binders, by the induction, with what is known there. A handler
   is entered at a static environment that extends that of its
   definition, and at a heap and a record that those have evolved to
   (FlambdaContextMemoryProofs.let_cont_handler_refines_where_memory). *)
Fixpoint share_primitives_knowing_refines (e : expr) :
  ∀ (known : computed) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    all_computed_or_made known vars statics hp rd ->
    refines (State e vars conts statics hp rd)
      (State (share_primitives_knowing known e) vars conts statics hp rd)
with share_primitives_in_handler_refines (h : continuation_handler) :
  ∀ (known : computed) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    all_computed_or_made known vars statics hp rd ->
    refines (State (Let_cont_non_recursive k h body) vars conts statics hp rd)
      (State
         (Let_cont_non_recursive k (share_primitives_in_handler known h)
            body)
         vars conts statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known vars conts statics hp rd Hhold; simpl.
    + (* Let *)
      apply let_refines_step_forward.
      * intros v hp' rd' Hd.
        exact (share_named_sound known vars statics hp hp' rd rd' d v Hhold
          Hd).
      * intros v hp' rd' Hd _.
        apply share_primitives_knowing_refines.
        exact
          (all_computed_or_made_under_let known x d vars statics hp hp' rd
             rd' v Hhold Hd).
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply share_primitives_knowing_refines.
      exact
        (all_computed_or_made_extend known vars statics extended hp rd Hhold
           (eval_static_bindings_extend vars bindings _ _ Hbindings)).
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k (share_primitives_in_handler known h)
              body)
           vars conts statics hp rd)).
      * apply share_primitives_in_handler_refines, Hhold.
      * apply let_cont_non_recursive_refines_step,
          share_primitives_knowing_refines, Hhold.
    + (* Let_cont_recursive *)
      apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply share_primitives_knowing_refines, Hhold.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros known k body vars conts statics hp rd Hhold. simpl.
    apply
      (let_cont_handler_refines_where_memory
         (all_computed_or_made known vars)
         (all_computed_or_made_extend known vars)
         (all_computed_or_made_evolves known vars));
      [ exact Hhold | ].
    intros vs handler_vars conts' statics' hp' rd' Hhold' _ Hbind.
    apply share_primitives_knowing_refines.
    exact
      (all_computed_or_made_bind_params statics' hp' rd' params known vs
         vars handler_vars Hhold' Hbind).
Qed.

Theorem share_known_primitives_refines :
  share_known_primitives_refines_statement.
Proof.
  intros e vars conts statics hp rd. apply share_primitives_knowing_refines.
  intros entry [].
Qed.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

(* The laws of this section are proved from the definitions
   ([equivariance], of FlambdaEquivarianceTactic.v), but for three
   things that the tactic is given.

   The test of two lists of simples ([same_simples]) is made by a
   decision procedure on lists, of which the database has no law: its
   law is proved by hand, from the injectivity of renaming, and given
   to the database.

   Computation is kept from unfolding [var_in_simples] and
   [same_simples] here, so that their laws apply where the tactic
   meets them in a test.

   The law of [forget_computed] is not left to the tactic from its
   start: the two definitions are unfolded and the database rewrites
   the filter of a mapped list, one step by hand goes under the [map]
   that both sides then begin with, and the tactic closes what is
   left. The tactic does not go under it itself: the type of the
   entries is written in two ways on the two sides ([variable] in
   [computed], the type that it is defined as in the test of the
   definition).

   A law that another file uses by its name is stated; a law that is
   stated is given to the database, and is not derived again. *)

Lemma var_in_simples_renamed (r : renaming) (x : variable)
    (args : list simple) :
  var_in_simples (rename_variable r x) (map (rename_simple r) args)
  = var_in_simples x args.
Proof. equivariance. Qed.

Lemma rename_simple_injective (r : renaming) (s s' : simple) :
  rename_simple r s = rename_simple r s' -> s = s'.
Proof.
  destruct s as [x | sym | c], s' as [x' | sym' | c']; simpl; intros H;
    inversion H; try reflexivity.
  f_equal. exact (permute_injective (variable_permutation r) x x' H1).
Qed.

Lemma rename_simples_injective (r : renaming) (args : list simple) :
  ∀ args' : list simple,
    map (rename_simple r) args = map (rename_simple r) args' -> args = args'.
Proof.
  induction args as [ | s args IH]; intros [ | s' args']; simpl; intros H;
    try discriminate H; [ reflexivity | ].
  inversion H as [[Hs Hargs]].
  rewrite (rename_simple_injective r s s' Hs), (IH args' Hargs).
  reflexivity.
Qed.

Lemma same_simples_renamed (r : renaming) (args args' : list simple) :
  same_simples (map (rename_simple r) args) (map (rename_simple r) args')
  = same_simples args args'.
Proof.
  unfold same_simples.
  destruct (list_eq_dec simple_eq_dec args args') as [-> | Hdiffer].
  - destruct
      (list_eq_dec simple_eq_dec (map (rename_simple r) args')
         (map (rename_simple r) args'))
      as [_ | Hne];
      [ reflexivity | contradiction Hne; reflexivity ].
  - destruct
      (list_eq_dec simple_eq_dec (map (rename_simple r) args)
         (map (rename_simple r) args'))
      as [Hsame | _];
      [ | reflexivity ].
    contradiction Hdiffer.
    exact (rename_simples_injective r args args' Hsame).
Qed.

#[local] Arguments var_in_simples : simpl never.
#[local] Arguments same_simples : simpl never.
#[export] Hint Rewrite same_simples_renamed : equivariance.

Lemma computed_by_renamed (r : renaming) (known : computed) (p : prim)
    (args : list simple) :
  computed_by (rename_computed r known) p (map (rename_simple r) args)
  = option_map (rename_variable r) (computed_by known p args).
Proof. equivariance. Qed.
#[export] Hint Rewrite computed_by_renamed : equivariance.

Lemma made_length_named_renamed (r : renaming) (known : computed) (d : named) :
  made_length_named (rename_computed r known) (rename_named r d)
  = option_map (rename_named r) (made_length_named known d).
Proof. equivariance. Qed.
#[export] Hint Rewrite made_length_named_renamed : equivariance.

Lemma share_named_renamed (r : renaming) (known : computed) (d : named) :
  share_named (rename_computed r known) (rename_named r d)
  = rename_named r (share_named known d).
Proof. equivariance. Qed.
#[export] Hint Rewrite share_named_renamed : equivariance.

Lemma forget_computed_renamed (r : renaming) (x : variable)
    (known : computed) :
  forget_computed (rename_variable r x) (rename_computed r known)
  = rename_computed r (forget_computed x known).
Proof.
  unfold forget_computed, rename_computed.
  autorewrite with equivariance. f_equal. eqv_core.
Qed.
#[export] Hint Rewrite forget_computed_renamed : equivariance.

Lemma forget_computed_vars_renamed (r : renaming) (xs : list variable) :
  ∀ known : computed,
    forget_computed_vars (map (rename_variable r) xs)
      (rename_computed r known)
    = rename_computed r (forget_computed_vars xs known).
Proof. equivariance. Qed.
#[export] Hint Rewrite forget_computed_vars_renamed : equivariance.

Lemma computed_under_let_renamed (r : renaming) (known : computed)
    (x : variable) (d : named) :
  computed_under_let (rename_computed r known) (rename_variable r x)
    (rename_named r d)
  = rename_computed r (computed_under_let known x d).
Proof. equivariance. Qed.
#[export] Hint Rewrite computed_under_let_renamed : equivariance.

Lemma array_made_under_let_renamed (r : renaming) (x : variable) (d : named) :
  array_made_under_let (rename_variable r x) (rename_named r d)
  = rename_computed r (array_made_under_let x d).
Proof. equivariance. Qed.
#[export] Hint Rewrite array_made_under_let_renamed : equivariance.

Lemma computed_or_made_under_let_renamed (r : renaming) (known : computed)
    (x : variable) (d : named) :
  computed_or_made_under_let (rename_computed r known) (rename_variable r x)
    (rename_named r d)
  = rename_computed r (computed_or_made_under_let known x d).
Proof. equivariance. Qed.
#[export] Hint Rewrite computed_or_made_under_let_renamed : equivariance.

Theorem share_primitives_knowing_equivariant :
  share_primitives_knowing_equivariant_statement.
Proof. equivariance. Qed.

Theorem share_known_primitives_equivariant :
  share_known_primitives_equivariant_statement.
Proof. intros r e. exact (share_primitives_knowing_equivariant r [] e). Qed.
