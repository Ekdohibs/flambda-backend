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


(* Proofs of the statements of FlambdaUnboxParamsSpec.v. Nothing here
   needs review.

   Giving a continuation one more parameter changes its handler and
   every application of it together, so it is proved by a simulation
   ([state_rel]) and not part by part ([One_parameter]). The simulation
   is one in which a step of the second state is answered by one or
   more of the first, or by a refinement that is already known
   ([Ahead]): at an application of the continuation the second run
   enters the handler after, and that this refines the handler before,
   entered with the same values, is asked as a premise
   ([handlers_alike]), with what the value of the new argument must
   be. The premise holds where
   the handler after is the handler before without a [Let] that reads
   a parameter ([read_refines]): by the lemmas of contexts, down to
   the [Let], which binds its variable to what the new parameter holds
   already. The pass is then that rewrite, for the [Let] that a handler
   begins with, done again and again, each time under the facts that
   hold where its [Let_cont] is. FlambdaUnboxVariantsProofs.v uses the
   same two for a [Let] further down.

   For a recursive continuation that is alone in its [Let_cont] the
   handler after applies the continuation itself, so the two runs go
   on together from the handler. What is asked there ([enters_loop])
   is that the handler before, entered with the values of an
   application, comes in one step to an expression of which the
   handler after is the extension, with the new parameter bound
   ([unbox_loop_parameter_refines]). FlambdaUnboxLoopsProofs.v uses it
   for a read that is first taken to the head of the handler. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaBlockLoadSpec FlambdaBlockLoadProofs FlambdaUnboxSpec
  FlambdaUnboxProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaCseSpec FlambdaCseProofs FlambdaUnboxParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Simulations in which the first run takes one step or more for each       *)
(* step of the second                                                       *)
(* ======================================================================== *)

Section Ahead.

Variable R : state -> state -> Prop.

(* What is asked of two related states: the first goes wrong; or both
   exit alike at once; or the second takes a step, and the first one or
   more, to related states; or the second is known to refine the
   first. *)
Definition answers (s s' : state) : Prop :=
  (∃ n : nat, run n s = Undefined)
  ∨ (∃ (k : continuation) (vs : list value) (hp : heap),
       step s = Exit k vs hp ∧ step s' = Exit k vs hp)
  ∨ (∃ (n : nat) (t t' : state),
       run (S n) s = Step t ∧ step s' = Step t' ∧ R t t')
  ∨ refines s s'.

Hypothesis R_step : ∀ s s' : state, R s s' -> answers s s'.

(* An exit of the first run is one of the second. *)
Lemma ahead_exit (n : nat) :
  ∀ (s s' : state) (k : continuation) (vs : list value) (hp : heap),
    R s s' ->
    run n s = Exit k vs hp ->
    ∃ m : nat, run m s' = Exit k vs hp.
Proof.
  induction n as [n IH] using (well_founded_induction lt_wf).
  intros s s' k vs hp Hrel Hexit.
  assert (Hs : behaves s (Exits k vs hp)) by (exists n; exact Hexit).
  destruct (R_step s s' Hrel)
    as [[n0 Hwrong] | [(k1 & vs1 & hp1 & Hstep & Hstep')
                      | [(j & t & t' & Hrun & Hstep' & Hrel') | Hrefines]]];
    [ | | | exact (proj1 Hrefines k vs hp Hs) ].
  - assert (Hs' : behaves s Goes_wrong) by (exists n0; exact Hwrong).
    discriminate (behaviour_unique s _ _ Hs Hs').
  - assert (Hs' : behaves s (Exits k1 vs1 hp1))
      by (exists 1; simpl; rewrite Hstep; reflexivity).
    injection (behaviour_unique s _ _ Hs Hs') as <- <- <-.
    exists 1. simpl. rewrite Hstep'. reflexivity.
  - destruct (Nat.lt_ge_cases n (S j)) as [Hshort | Hlong].
    + exfalso. replace (S j) with (n + (S j - n)) in Hrun by lia.
      rewrite (ended_stable n _ s (Exits k vs hp)) in Hrun
        by (rewrite Hexit; simpl; auto).
      rewrite Hexit in Hrun. discriminate.
    + replace n with (S j + (n - S j)) in Hexit by lia.
      rewrite run_add, Hrun in Hexit.
      assert (Hless : n - S j < n) by lia.
      destruct (IH (n - S j) Hless t t' k vs hp Hrel' Hexit) as [m Hm].
      exists (S m). simpl. rewrite Hstep'. exact Hm.
Qed.

(* If the second run has ended, the first has ended alike or gone
   wrong. *)
Lemma ahead_ends (m : nat) :
  ∀ (s s' : state) (b : behaviour),
    R s s' ->
    ended (run m s') b ->
    behaves s b ∨ behaves s Goes_wrong.
Proof.
  induction m as [ | m IH]; intros s s' b Hrel Hended;
    [ destruct b; contradiction | ].
  destruct (R_step s s' Hrel)
    as [[n0 Hwrong] | [(k1 & vs1 & hp1 & Hstep & Hstep')
                      | [(j & t & t' & Hrun & Hstep' & Hrel') | Hrefines]]].
  - right. exists n0. exact Hwrong.
  - left. simpl in Hended. rewrite Hstep' in Hended.
    destruct b as [k vs hp | | ]; try contradiction.
    destruct Hended as [<- [<- <-]].
    exists 1. simpl. rewrite Hstep. reflexivity.
  - simpl in Hended. rewrite Hstep' in Hended.
    destruct (IH t t' b Hrel' Hended) as [Ht | Ht].
    + left. apply (run_behaves (S j) s b). rewrite Hrun. exact Ht.
    + right. apply (run_behaves (S j) s Goes_wrong). rewrite Hrun. exact Ht.
  - apply (proj2 Hrefines b); [ | exact (ended_behaves (S m) s' b Hended) ].
    intros ->. destruct (run (S m) s'); contradiction.
Qed.

Lemma ahead_refines (s s' : state) : R s s' -> refines s s'.
Proof.
  intros Hrel. split.
  - intros k vs hp [n Hn]. exact (ahead_exit n s s' k vs _ Hrel Hn).
  - intros b Hterm Hb. destruct (behaves_ended s' b Hterm Hb) as [m Hended].
    exact (ahead_ends m s s' b Hrel Hended).
Qed.

End Ahead.

(* ======================================================================== *)
(* What is known holds                                                      *)
(* ======================================================================== *)

(* The fields of a block that a [Let] in scope made are of the kind of
   values: [Make_block] has no value otherwise. *)
Definition fields_valued (known : known_blocks) (vars : env) : Prop :=
  ∀ (y : variable) (block : known_block) (tag : Z) (vs : list value),
    (y, block) ∈ known_vars known ->
    lookup y vars = Some (Value_block tag vs) ->
    Forall (fun v => kind_of_value v = Value) vs.

(* The number in a box that a [Let] in scope made is of the kind of
   the box: [Box_number] has no value otherwise. *)
Definition numbers_kinded (known : known_numbers) (vars : env) : Prop :=
  ∀ (y : variable) (kind : boxable_number) (contents : simple) (c : const),
    (y, (Boxed kind, contents)) ∈ known_wrapped_vars known ->
    lookup y vars = Some (Value_boxed_number c) ->
    is_number_of_kind kind c = true.

Definition facts_hold (known : facts) (vars : env) (statics : static_env)
    : Prop :=
  blocks_hold (facts_blocks known) vars statics
  ∧ numbers_hold (facts_numbers known) vars statics
  ∧ fields_valued (facts_blocks known) vars
  ∧ numbers_kinded (facts_numbers known) vars
  ∧ all_computed (facts_reads known) vars statics.

Lemma no_facts_hold (vars : env) (statics : static_env) :
  facts_hold no_facts vars statics.
Proof.
  split; [ split; intros entry [] | ].
  split; [ split; intros entry [] | ].
  split; [ intros y block tag vs [] | ].
  split; [ intros y kind contents c [] | intros entry [] ].
Qed.

Lemma fields_valued_forget_var (known : known_blocks) (vars : env)
    (x : variable) (v : value) :
  fields_valued known vars ->
  fields_valued (forget_var x known) ((x, v) :: vars).
Proof.
  intros Hvalued y block tag vs Hin Hlookup. simpl in Hin.
  apply in_map_iff in Hin. destruct Hin as ([y0 block0] & Heq & Hin).
  apply filter_In in Hin. destruct Hin as [Hin Hkey].
  simpl in Heq, Hkey. injection Heq as -> _.
  apply negb_true_iff, Pos.eqb_neq in Hkey.
  rewrite lookup_other in Hlookup by congruence.
  exact (Hvalued y block0 tag vs Hin Hlookup).
Qed.

Lemma fields_valued_under_let (known : known_blocks) (x : variable)
    (d : named) (vars : env) (statics : static_env) (v : value) :
  fields_valued known vars ->
  eval_named statics vars d = Some v ->
  fields_valued (known_under_let known x d) ((x, v) :: vars).
Proof.
  intros Hvalued Hd. unfold known_under_let.
  pose proof (fields_valued_forget_var known vars x v Hvalued) as Hforgot.
  destruct (block_made d) as [made | ] eqn:Hmade; [ | exact Hforgot ].
  intros y block tag vs Hin Hlookup. simpl in Hin.
  destruct Hin as [Heq | Hin]; [ | exact (Hforgot y block tag vs Hin Hlookup) ].
  injection Heq as <- _. rewrite lookup_head in Hlookup.
  injection Hlookup as ->.
  destruct (block_made_spec d made Hmade) as (fields & -> & _).
  simpl in Hd.
  destruct (eval_simples vars fields) as [vs0 | ]; [ | discriminate ].
  simpl in Hd. rewrite eval_prim_variadic in Hd. simpl in Hd.
  destruct (forallb (fun v => kind_eqb (kind_of_value v) Value) vs0)
    eqn:Hkinds; [ | discriminate ].
  destruct (forallb (bound_if_symbol statics) vs0); [ | discriminate ].
  injection Hd as _ <-. apply Forall_forall. intros u Hu.
  apply kind_eqb_eq. exact (proj1 (forallb_forall _ _) Hkinds u Hu).
Qed.

Lemma fields_valued_bind_params (params : list variable) :
  ∀ (known : known_blocks) (vs : list value) (vars handler_vars : env),
    fields_valued known vars ->
    bind_params params vs vars = Some handler_vars ->
    fields_valued (forget_vars params known) handler_vars.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hvalued; simpl;
    try discriminate.
  - intros [= <-]. exact Hvalued.
  - apply IH, fields_valued_forget_var, Hvalued.
Qed.

Lemma numbers_kinded_forget_var (known : known_numbers) (vars : env)
    (x : variable) (v : value) :
  numbers_kinded known vars ->
  numbers_kinded (forget_wrapped_var x known) ((x, v) :: vars).
Proof.
  intros Hkinded y kind contents c Hin Hlookup.
  unfold forget_wrapped_var in Hin. cbn [known_wrapped_vars] in Hin.
  apply filter_In in Hin. destruct Hin as [Hin Hkept]. cbv beta in Hkept.
  apply andb_true_iff in Hkept as [Hkey _]. simpl in Hkey.
  apply negb_true_iff, Pos.eqb_neq in Hkey.
  rewrite lookup_other in Hlookup by congruence.
  exact (Hkinded y kind contents c Hin Hlookup).
Qed.

Lemma numbers_kinded_under_let (known : known_numbers) (x : variable)
    (d : named) (vars : env) (statics : static_env) (v : value) :
  numbers_kinded known vars ->
  eval_named statics vars d = Some v ->
  numbers_kinded (numbers_under_let known x d) ((x, v) :: vars).
Proof.
  intros Hkinded Hd. unfold numbers_under_let.
  pose proof (numbers_kinded_forget_var known vars x v Hkinded) as Hforgot.
  destruct (number_wrapped d) as [[how wrapped] | ] eqn:Hwrapped;
    [ | exact Hforgot ].
  destruct (var_in_simple x wrapped); [ exact Hforgot | ].
  intros y kind contents c Hin Hlookup. cbn [known_wrapped_vars] in Hin.
  destruct Hin as [Heq | Hin];
    [ | exact (Hforgot y kind contents c Hin Hlookup) ].
  injection Heq as <- -> <-. rewrite lookup_head in Hlookup.
  injection Hlookup as ->.
  destruct (number_wrapped_spec d (Boxed kind) wrapped Hwrapped)
    as [(kind' & [= <-] & ->) | [Hhow _]]; [ | discriminate Hhow ].
  simpl in Hd.
  destruct (eval_simple vars wrapped) as [unwrapped | ]; [ | discriminate ].
  simpl in Hd.
  destruct (eval_box_number statics kind unwrapped _ Hd)
    as (c0 & -> & [= <-]).
  rewrite eval_prim_unary in Hd. simpl in Hd.
  destruct (is_number_of_kind kind c) eqn:Hkind;
    [ reflexivity | discriminate ].
Qed.

Lemma numbers_kinded_bind_params (params : list variable) :
  ∀ (known : known_numbers) (vs : list value) (vars handler_vars : env),
    numbers_kinded known vars ->
    bind_params params vs vars = Some handler_vars ->
    numbers_kinded (forget_wrapped_vars params known) handler_vars.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hkinded; simpl;
    try discriminate.
  - intros [= <-]. exact Hkinded.
  - apply IH, numbers_kinded_forget_var, Hkinded.
Qed.

(* A read that a [Let] makes is known in its body: the variable is
   bound to the value of the primitive on the value of the variable
   read, which the [Let] does not bind. *)
Lemma reads_hold_under_let (known : computed) (x : variable) (d : named)
    (vars : env) (statics : static_env) (v : value) :
  all_computed known vars statics ->
  eval_named statics vars d = Some v ->
  all_computed (reads_under_let known x d) ((x, v) :: vars) statics.
Proof.
  intros Hhold Hd. unfold reads_under_let.
  pose proof (all_computed_forget known vars statics x v Hhold) as Hforgot.
  destruct d as [s | p args | | | | | ]; try exact Hforgot.
  destruct p as [p | | | | ]; try exact Hforgot.
  destruct args as [ | [y | sym | c] [ | arg args]]; try exact Hforgot.
  destruct (reads_a_box p && negb (Pos.eqb x y)) eqn:Hread;
    [ | exact Hforgot ].
  apply andb_true_iff in Hread as [_ Hne].
  apply negb_true_iff, Pos.eqb_neq in Hne.
  intros entry [<- | Hin]; [ | exact (Hforgot entry Hin) ].
  simpl in Hd.
  destruct (lookup y vars) as [w | ] eqn:Hy; [ | discriminate ].
  simpl in Hd.
  exists [w], v. cbn [fst snd]. split; [ | split; [ exact Hd | ] ].
  - assert (Hlookup : lookup y ((x, v) :: vars) = Some w)
      by (rewrite lookup_other by congruence; exact Hy).
    simpl. simpl in Hlookup. rewrite Hlookup. reflexivity.
  - apply lookup_head.
Qed.

Lemma facts_hold_under_let (known : facts) (x : variable) (d : named)
    (vars : env) (statics : static_env) (v : value) :
  facts_hold known vars statics ->
  eval_named statics vars d = Some v ->
  facts_hold (facts_under_let known x d) ((x, v) :: vars) statics.
Proof.
  intros (Hblocks & Hnumbers & Hvalued & Hkinded & Hreads) Hd.
  split; [ | split; [ | split; [ | split ] ] ].
  - exact (blocks_hold_under_let _ x d vars statics v Hblocks Hd).
  - exact (numbers_hold_under_let _ x d vars statics v Hnumbers Hd).
  - exact (fields_valued_under_let _ x d vars statics v Hvalued Hd).
  - exact (numbers_kinded_under_let _ x d vars statics v Hkinded Hd).
  - exact (reads_hold_under_let _ x d vars statics v Hreads Hd).
Qed.

Lemma facts_hold_under_statics (known : facts) (vars : env)
    (statics : static_env) (bindings : list static_binding)
    (extended : static_env) :
  facts_hold known vars statics ->
  eval_static_bindings vars bindings statics = Some extended ->
  facts_hold (facts_under_statics known bindings) vars extended.
Proof.
  intros (Hblocks & Hnumbers & Hvalued & Hkinded & Hreads) Heval.
  split; [ | split; [ | split; [ | split ] ] ].
  - exact (blocks_hold_under_statics _ vars statics bindings extended
      Hblocks Heval).
  - exact (numbers_hold_under_statics _ vars statics bindings extended
      Hnumbers Heval).
  - exact Hvalued.
  - exact Hkinded.
  - exact (all_computed_extend _ vars statics extended Hreads
      (eval_static_bindings_extend vars bindings _ _ Heval)).
Qed.

Lemma facts_hold_bind_params (statics : static_env)
    (params : list variable) (known : facts) (vs : list value)
    (vars handler_vars : env) :
  facts_hold known vars statics ->
  bind_params params vs vars = Some handler_vars ->
  facts_hold (facts_under_params known params) handler_vars statics.
Proof.
  intros (Hblocks & Hnumbers & Hvalued & Hkinded & Hreads) Hbind.
  split; [ | split; [ | split; [ | split ] ] ].
  - exact (blocks_hold_bind_params statics params _ vs vars handler_vars
      Hblocks Hbind).
  - exact (numbers_hold_bind_params statics params _ vs vars handler_vars
      Hnumbers Hbind).
  - exact (fields_valued_bind_params params _ vs vars handler_vars Hvalued
      Hbind).
  - exact (numbers_kinded_bind_params params _ vs vars handler_vars Hkinded
      Hbind).
  - exact (all_computed_bind_params statics params _ vs vars handler_vars
      Hreads Hbind).
Qed.

Lemma facts_hold_extend (known : facts) (vars : env)
    (statics statics' : static_env) :
  facts_hold known vars statics ->
  statics_extend statics statics' ->
  facts_hold known vars statics'.
Proof.
  intros ([Hblock_vars Hblock_symbols] & [Hnumber_vars Hnumber_symbols]
          & Hvalued & Hkinded & Hreads) Hmore.
  split; [ split | split; [ split | split; [ | split ] ] ].
  - exact Hblock_vars.
  - exact (symbols_hold_extend _ vars statics statics' Hblock_symbols Hmore).
  - exact Hnumber_vars.
  - exact (symbol_numbers_hold_extend _ statics statics' Hnumber_symbols
      Hmore).
  - exact Hvalued.
  - exact Hkinded.
  - exact (all_computed_extend _ vars statics statics' Hreads Hmore).
Qed.

(* ======================================================================== *)
(* What reading a box gives                                                 *)
(* ======================================================================== *)

(* The variable that a [Let] in scope bound to the primitive applied to
   the argument holds the value of the primitive on the value of the
   argument. *)
Lemma read_in_scope_sound (known : computed) (vars : env)
    (statics : static_env) (p : unary_primitive) (arg : simple)
    (x : variable) (v w : value) :
  all_computed known vars statics ->
  computed_by known (Unary p) [arg] = Some x ->
  eval_simple vars arg = Some v ->
  eval_prim statics (Unary p) [v] = Some w ->
  eval_simple vars (Var x) = Some w.
Proof.
  intros Hhold Hby Harg Hprim.
  destruct (Hhold _ (computed_by_sound _ _ _ _ Hby))
    as (vs & r & Hvs & Hr & Hlookup).
  cbn [fst snd] in Hvs, Hr, Hlookup. simpl in Hvs. rewrite Harg in Hvs.
  simpl in Hvs. injection Hvs as <-. rewrite Hprim in Hr.
  injection Hr as <-. exact Hlookup.
Qed.

(* Where the primitive has a value on the argument, it is the value of
   what [projection_of] gives. *)
Lemma projection_of_sound (known : facts) (vars : env)
    (statics : static_env) (p : unary_primitive) (arg projection : simple)
    (v w : value) :
  facts_hold known vars statics ->
  projection_of known p arg = Some projection ->
  eval_simple vars arg = Some v ->
  eval_prim statics (Unary p) [v] = Some w ->
  eval_simple vars projection = Some w.
Proof.
  intros (Hblocks & Hnumbers & _ & _ & Hreads) Hprojection Harg Hprim.
  assert (Hd : eval_named statics vars (Prim (Unary p) [arg]) = Some w)
    by (simpl; rewrite Harg; simpl; exact Hprim).
  assert (Hread :
    ∀ x : variable,
      computed_by (facts_reads known) (Unary p) [arg] = Some x ->
      eval_simple vars (Var x) = Some w).
  { intros x Hby.
    exact (read_in_scope_sound _ vars statics p arg x v w Hreads Hby Harg
      Hprim). }
  unfold projection_of in Hprojection.
  destruct (reads_a_box p); [ | discriminate ].
  pose proof (fold_named_sound statics vars (Prim (Unary p) [arg])) as Hfold.
  pose proof (load_known_named_sound (facts_blocks known) vars statics _ w
    Hblocks Hd) as Hload.
  pose proof (unbox_known_named_sound (facts_numbers known) vars statics _ w
    Hnumbers Hd) as Hunbox.
  destruct (fold_named (Prim (Unary p) [arg]));
    [ injection Hprojection as <-; rewrite <- Hd, <- Hfold; reflexivity
    | .. ];
    (destruct (load_known_named (facts_blocks known) (Prim (Unary p) [arg]));
     [ injection Hprojection as <-; exact Hload | .. ];
     (destruct
        (unbox_known_named (facts_numbers known) (Prim (Unary p) [arg]));
      [ injection Hprojection as <-; exact Hunbox | .. ];
      (destruct (computed_by (facts_reads known) (Unary p) [arg])
         as [x | ] eqn:Hby;
       [ injection Hprojection as <-;
         first [ exact (Hread x Hby) | exact (Hread x eq_refl) ]
       | discriminate ]))).
Qed.

(* ======================================================================== *)
(* Arguments and parameters                                                 *)
(* ======================================================================== *)

Lemma eval_simples_nth (vars : env) (args : list simple) :
  ∀ (vs : list value) (i : nat) (arg : simple),
    eval_simples vars args = Some vs ->
    nth_error args i = Some arg ->
    ∃ v : value, nth_error vs i = Some v ∧ eval_simple vars arg = Some v.
Proof.
  induction args as [ | first args IH]; intros vs i arg; simpl.
  - destruct i; discriminate.
  - destruct (eval_simple vars first) as [v | ] eqn:Hfirst; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. destruct i as [ | i]; simpl.
    + intros [= <-]. exists v. auto.
    + intros Hnth. exact (IH rest i arg eq_refl Hnth).
Qed.

Lemma eval_simples_snoc (vars : env) (args : list simple) (last : simple)
    (w : value) :
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    eval_simple vars last = Some w ->
    eval_simples vars (args ++ [last]) = Some (vs ++ [w]).
Proof.
  induction args as [ | first args IH]; intros vs; simpl.
  - intros [= <-] Hlast. rewrite Hlast. reflexivity.
  - destruct (eval_simple vars first) as [v | ]; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-] Hlast. rewrite (IH rest eq_refl Hlast). reflexivity.
Qed.

(* One more parameter, bound last, is the first of the variables. *)
Lemma bind_params_snoc (params : list variable) (f : variable) (w : value) :
  ∀ (vs : list value) (vars bound : env),
    bind_params params vs vars = Some bound ->
    bind_params (params ++ [f]) (vs ++ [w]) vars = Some ((f, w) :: bound).
Proof.
  induction params as [ | param params IH]; intros [ | v vs] vars bound;
    simpl; try discriminate.
  - intros [= <-]. reflexivity.
  - apply IH.
Qed.

(* A name that is not a parameter is bound as it was. *)
Lemma bind_params_other (params : list variable) (x : variable) :
  ∀ (vs : list value) (vars bound : env),
    among x params = false ->
    bind_params params vs vars = Some bound ->
    lookup x bound = lookup x vars.
Proof.
  induction params as [ | param params IH]; intros [ | v vs] vars bound;
    simpl; try discriminate.
  - intros _ [= <-]. reflexivity.
  - intros Hamong Hbind. apply orb_false_iff in Hamong as [Hne Hamong].
    rewrite (IH vs _ bound Hamong Hbind). simpl. rewrite Hne. reflexivity.
Qed.

(* Distinct parameters are bound each to its argument. *)
Lemma bind_params_nth (params : list variable) :
  ∀ (vs : list value) (vars bound : env) (i : nat) (x : variable)
    (v : value),
    distinct params = true ->
    bind_params params vs vars = Some bound ->
    nth_error params i = Some x ->
    nth_error vs i = Some v ->
    lookup x bound = Some v.
Proof.
  induction params as [ | param params IH]; intros [ | v0 vs] vars bound i x v;
    simpl; try discriminate; try (destruct i; discriminate).
  intros Hdistinct Hbind. apply andb_true_iff in Hdistinct as [Hfresh Hrest].
  apply negb_true_iff in Hfresh.
  destruct i as [ | i]; simpl.
  - intros [= <-] [= <-].
    rewrite (bind_params_other params param vs _ bound Hfresh Hbind).
    apply lookup_head.
  - exact (IH vs _ bound i x v Hrest Hbind).
Qed.

(* The kinds of values with one more, against kinds with one more. *)
Lemma kinds_snoc (vs : list value) (w : value) (kinds : list flambda_kind)
    (kind : flambda_kind) :
  kinds_eqb (map kind_of_value vs) kinds = true ->
  kind_eqb (kind_of_value w) kind = true ->
  kinds_eqb (map kind_of_value (vs ++ [w])) (kinds ++ [kind]) = true.
Proof.
  intros Hkinds Hkind. apply kinds_eqb_eq in Hkinds.
  apply kind_eqb_eq in Hkind.
  rewrite map_app, Hkinds. simpl. rewrite Hkind. apply kinds_eqb_refl.
Qed.

(* A parameter that is bound has an argument. *)
Lemma bind_params_value (params : list variable) :
  ∀ (vs : list value) (vars bound : env) (i : nat) (x : variable),
    bind_params params vs vars = Some bound ->
    nth_error params i = Some x ->
    ∃ v : value, nth_error vs i = Some v.
Proof.
  induction params as [ | param params IH]; intros [ | v0 vs] vars bound i x;
    simpl; try discriminate; try (destruct i; discriminate).
  intros Hbind. destruct i as [ | i]; simpl.
  - intros _. exists v0. reflexivity.
  - exact (IH vs _ bound i x Hbind).
Qed.

Lemma position_among (x : variable) (xs : list variable) :
  ∀ i : nat, position x xs = Some i -> among x xs = true.
Proof.
  induction xs as [ | y xs IH]; intros i; simpl; [ discriminate | ].
  destruct (Pos.eqb x y); [ reflexivity | ].
  destruct (position x xs) as [j | ]; [ | discriminate ].
  intros _. exact (IH j eq_refl).
Qed.

Lemma position_nth (x : variable) (xs : list variable) :
  ∀ i : nat, position x xs = Some i -> nth_error xs i = Some x.
Proof.
  induction xs as [ | y xs IH]; intros i; simpl; [ discriminate | ].
  destruct (Pos.eqb_spec x y) as [-> | _].
  - intros [= <-]. reflexivity.
  - destruct (position x xs) as [j | ]; [ | discriminate ].
    intros [= <-]. exact (IH j eq_refl).
Qed.

(* ======================================================================== *)
(* One [Let] of a handler made a parameter                                  *)
(* ======================================================================== *)

Section One_parameter.

(* The continuation, the position of the parameter that is read, the
   new parameter and its kind, the handler before and after, what
   gives an application its new argument, and what is asked of the
   value of that argument, given the values of the others. *)
Variable k : continuation.
Variable i : nat.
Variable params : list variable.
Variable metadata metadata' : handler_metadata.
Variable f x : variable.
Variable kind : flambda_kind.
Variable boxed unboxed : expr.
Variable extend : facts -> list simple -> option (list simple).
Variable fits : static_env -> list value -> value -> Prop.

Hypothesis params_distinct : distinct params = true.
Hypothesis x_at_i : nth_error params i = Some x.
Hypothesis kinds_one_more :
  hm_params_kinds metadata' = hm_params_kinds metadata ++ [kind].

(* The new argument of an application has a value of the kind of the
   new parameter, which is as asked; or the handler before, entered
   with that value of the parameter read, goes wrong. *)
Hypothesis extend_gives :
  ∀ (known : facts) (vars : env) (statics : static_env) (hp : heap)
    (args args' : list simple) (vs : list value) (v : value),
    facts_hold known vars statics ->
    extend known args = Some args' ->
    eval_simples vars args = Some vs ->
    nth_error vs i = Some v ->
    (∃ u : value,
       eval_simples vars args' = Some (vs ++ [u])
       ∧ kind_of_value u = kind
       ∧ fits statics vs u)
    ∨ (∀ (bound : env) (conts : cont_env),
         lookup x bound = Some v ->
         ∃ n : nat, run n (State boxed bound conts statics hp) = Undefined).

(* What is asked where the continuation is not recursive: the handler
   after, entered with a new argument that is as asked, refines the
   handler before. *)
Definition handlers_alike : Prop :=
  ∀ (def_vars bound : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (vs : list value) (u : value),
    bind_params params vs def_vars = Some bound ->
    fits statics vs u ->
    refines (State boxed bound conts statics hp)
      (State unboxed ((f, u) :: bound) conts statics hp).

(* What is asked where the continuation is recursive, alone in its
   [Let_cont], which is defined where [known] holds: the handler
   before, entered with the values of an application, goes wrong, or
   comes in one step, with the new parameter bound to an argument that
   is as asked, to an expression of which the handler after is the
   extension, with what is known there. The applications in the
   handler are of the continuation itself, so the two runs go on in
   step from there. *)
Definition enters_loop (known : facts) : Prop :=
  ∀ (def_vars bound : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (vs : list value) (u : value),
    facts_hold known def_vars statics ->
    bind_params params vs def_vars = Some bound ->
    fits statics vs u ->
    (∃ n : nat, run n (State boxed bound conts statics hp) = Undefined)
    ∨ (∃ (rest : expr) (inside : facts),
         step (State boxed bound conts statics hp)
         = Step (State rest ((f, u) :: bound) conts statics hp)
         ∧ extend_applications_by extend inside k rest = Some unboxed
         ∧ facts_hold inside ((f, u) :: bound) statics).

(* The handlers of a recursive [Let_cont] with the applications in them
   extended. *)
Definition extended_handlers (known : facts)
    (handlers : list (continuation * continuation_handler))
    : option (list (continuation * continuation_handler)) :=
  all_some
    (map
       (fun '(k', h) =>
          let* h := extend_in_handler_by extend known k h in
          Some (k', h))
       handlers).

(* Two expressions, in the scope of the continuation or out of it: in
   it, the second is the first with its applications extended; out of
   it, they are the same. *)
Definition expr_rel (inside : bool) (known : facts) (e e' : expr) : Prop :=
  if inside then extend_applications_by extend known k e = Some e' else e' = e.

Definition handlers_rel (inside : bool) (known : facts)
    (handlers handlers' : list (continuation * continuation_handler))
    : Prop :=
  if inside then extended_handlers known handlers = Some handlers'
  else handlers' = handlers.

(* What a continuation name stands for in the two runs. For the name
   [k] where it is the continuation being unboxed ([true]): the two
   handlers, defined under the same environments, each with no
   recursive handlers, or each the one recursive handler of its own
   closure, defined where what is known holds. For any other
   ([false]): the same thing; or closures whose handlers are related,
   under the same variables, of which the facts that the handlers were
   extended with hold, and under continuations that are in turn
   related; or two that are related, with the same extra arguments; or
   what is left of the same over-application, to return and to raise
   to two that are related. The handlers of a closure are in the scope
   of the continuation if the closure was defined in it and is not one
   of a recursive [Let_cont] that binds the name again. *)
Inductive cont_rel (statics : static_env)
    : bool -> cont_value -> cont_value -> Prop :=
  | Same_rel (c : cont_value) : cont_rel statics false c c
  | Unboxed_rel (vars : env) (conts : cont_env) :
    handlers_alike ->
    cont_rel statics true
      (Closure (Handler params metadata boxed) vars conts [])
      (Closure (Handler (params ++ [f]) metadata' unboxed) vars conts [])
  | Closure_rel (inside : bool) (known : facts) (params0 : list variable)
      (metadata0 : handler_metadata) (handler handler' : expr) (vars : env)
      (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    handlers_rel (inside && negb (is_bound k recursive)) known recursive
      recursive' ->
    expr_rel (inside && negb (is_bound k recursive))
      (facts_under_params known params0) handler handler' ->
    facts_hold known vars statics ->
    (∀ k' : continuation,
      cont_rel statics (inside && Pos.eqb k' k) (resolve k' conts)
        (resolve k' conts')) ->
    cont_rel statics false
      (Closure (Handler params0 metadata0 handler) vars conts recursive)
      (Closure (Handler params0 metadata0 handler') vars conts' recursive')
  | Extra_args_rel (c c' : cont_value) (extra : list value) :
    cont_rel statics false c c' ->
    cont_rel statics false (Extra_args c extra) (Extra_args c' extra)
  | Over_application_rel (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    cont_rel statics false return_to return_to' ->
    cont_rel statics false exn_to exn_to' ->
    cont_rel statics false
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to')
  | Unboxed_loop_rel (known : facts) (vars : env) (conts : cont_env) :
    facts_hold known vars statics ->
    enters_loop known ->
    cont_rel statics true
      (Closure (Handler params metadata boxed) vars conts
         [(k, Handler params metadata boxed)])
      (Closure (Handler (params ++ [f]) metadata' unboxed) vars conts
         [(k, Handler (params ++ [f]) metadata' unboxed)]).

Definition conts_rel (statics : static_env) (inside : bool)
    (conts conts' : cont_env) : Prop :=
  ∀ k' : continuation,
    cont_rel statics (inside && Pos.eqb k' k) (resolve k' conts)
      (resolve k' conts').

Inductive state_rel : state -> state -> Prop :=
  | State_rel (inside : bool) (known : facts) (e e' : expr) (vars : env)
      (conts conts' : cont_env) (statics : static_env) (hp : heap) :
    expr_rel inside known e e' ->
    facts_hold known vars statics ->
    conts_rel statics inside conts conts' ->
    state_rel (State e vars conts statics hp) (State e' vars conts' statics hp).

Lemma state_rel_refl (s : state) : state_rel s s.
Proof.
  destruct s as [e vars conts statics].
  apply (State_rel false no_facts);
    [ reflexivity | apply no_facts_hold | intros k'; apply Same_rel ].
Qed.

(* A static environment that binds more. *)
Lemma cont_rel_extend (statics statics' : static_env) (inside : bool)
    (c c' : cont_value) :
  statics_extend statics statics' ->
  cont_rel statics inside c c' ->
  cont_rel statics' inside c c'.
Proof.
  intros Hmore Hc.
  induction Hc
    as [ c | vars conts Halike
       | inside known params0 metadata0 handler handler' vars conts conts'
           recursive recursive' Hrecursive Hhandler Hfacts Hconts IH
       | c c' extra Hc IH
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise IHraise
       | known vars conts Hfacts Henters ].
  - apply Same_rel.
  - apply Unboxed_rel, Halike.
  - apply (Closure_rel statics' inside known);
      [ exact Hrecursive | exact Hhandler
      | exact (facts_hold_extend _ _ _ _ Hfacts Hmore) | exact IH ].
  - apply Extra_args_rel, IH.
  - apply Over_application_rel; assumption.
  - apply (Unboxed_loop_rel statics' known);
      [ exact (facts_hold_extend _ _ _ _ Hfacts Hmore) | exact Henters ].
Qed.

Lemma conts_rel_extend (statics statics' : static_env) (inside : bool)
    (conts conts' : cont_env) :
  statics_extend statics statics' ->
  conts_rel statics inside conts conts' ->
  conts_rel statics' inside conts conts'.
Proof.
  intros Hmore Hconts k'. exact (cont_rel_extend _ _ _ _ _ Hmore (Hconts k')).
Qed.

Lemma with_extra_args_rel (statics : static_env) (c c' : cont_value)
    (extra : list value) :
  cont_rel statics false c c' ->
  cont_rel statics false (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; simpl; [ exact Hc | apply Extra_args_rel, Hc ].
Qed.

(* A continuation bound by a non-recursive [Let_cont]: the name [k]
   bound again is no longer the one being unboxed. *)
Lemma conts_rel_cons (statics : static_env) (inside : bool)
    (k0 : continuation) (c c' : cont_value) (conts conts' : cont_env) :
  cont_rel statics false c c' ->
  conts_rel statics inside conts conts' ->
  conts_rel statics (inside && negb (Pos.eqb k0 k)) ((k0, c) :: conts)
    ((k0, c') :: conts').
Proof.
  intros Hc Hconts k'. destruct (Pos.eq_dec k' k0) as [-> | Hne].
  - rewrite !resolve_head.
    destruct (Pos.eqb k0 k); rewrite ?andb_false_r; exact Hc.
  - rewrite !resolve_other by exact Hne. specialize (Hconts k').
    destruct (Pos.eqb_spec k0 k) as [Hk0 | _]; simpl.
    + rewrite andb_false_r. simpl.
      destruct (Pos.eqb_spec k' k) as [Hk' | _];
        [ exfalso; apply Hne; congruence | ].
      rewrite andb_false_r in Hconts. exact Hconts.
    + rewrite andb_true_r. exact Hconts.
Qed.

(* The handlers of a recursive [Let_cont], extended, one by one. *)
Lemma lookup_extended_handlers (known : facts) (k' : continuation) :
  ∀ handlers handlers' : list (continuation * continuation_handler),
    extended_handlers known handlers = Some handlers' ->
    match lookup k' handlers with
    | Some h =>
      ∃ h' : continuation_handler,
        extend_in_handler_by extend known k h = Some h'
        ∧ lookup k' handlers' = Some h'
    | None => lookup k' handlers' = None
    end.
Proof.
  unfold extended_handlers.
  induction handlers as [ | [k0 h] handlers IH]; intros handlers'; simpl.
  - intros [= <-]. reflexivity.
  - destruct (extend_in_handler_by extend known k h) as [h' | ] eqn:Hh;
      [ | discriminate ].
    destruct
      (all_some
         (map
            (fun '(k', h) =>
               let* h := extend_in_handler_by extend known k h in
               Some (k', h))
            handlers))
      as [rest_handlers | ]; [ | discriminate ].
    intros [= <-]. simpl. destruct (Pos.eqb k' k0).
    + exists h'. auto.
    + exact (IH rest_handlers eq_refl).
Qed.

(* The closures of the handlers of a recursive [Let_cont], over
   continuations that are related. *)
Lemma rec_conts_rel (statics : static_env) (inside : bool) (known : facts)
    (vars : env) (conts conts' : cont_env)
    (recursive recursive' : list (continuation * continuation_handler)) :
  handlers_rel (inside && negb (is_bound k recursive)) known recursive
    recursive' ->
  facts_hold known vars statics ->
  conts_rel statics inside conts conts' ->
  conts_rel statics (inside && negb (is_bound k recursive))
    (rec_closures recursive vars conts ++ conts)
    (rec_closures recursive' vars conts' ++ conts').
Proof.
  intros Hrecursive Hfacts Hconts k'. rewrite !resolve_rec_closures.
  pose proof (Closure_rel statics inside known) as Hclosure.
  remember (inside && negb (is_bound k recursive)) as scoped eqn:Hscoped.
  destruct scoped; unfold handlers_rel in Hrecursive.
  - symmetry in Hscoped. apply andb_true_iff in Hscoped as [-> Hunbound].
    apply negb_true_iff in Hunbound.
    pose proof (lookup_extended_handlers known k' _ _ Hrecursive) as Hlookup.
    destruct (lookup k' recursive) as [[params0 metadata0 handler] | ]
      eqn:Hfound.
    + destruct Hlookup as (h' & Hh & ->). simpl in Hh.
      destruct
        (extend_applications_by extend (facts_under_params known params0) k
           handler)
        as [handler' | ] eqn:Hhandler; [ | discriminate ].
      injection Hh as <-.
      assert (Hne : Pos.eqb k' k = false).
      { destruct (Pos.eqb k' k) eqn:Heq; [ | reflexivity ].
        apply Pos.eqb_eq in Heq. rewrite Heq in Hfound.
        unfold is_bound in Hunbound. rewrite Hfound in Hunbound.
        discriminate. }
      rewrite Hne. simpl.
      apply Hclosure; rewrite ?Hunbound; simpl;
        [ exact Hrecursive | exact Hhandler | exact Hfacts | exact Hconts ].
    + rewrite Hlookup. exact (Hconts k').
  - subst recursive'.
    destruct (lookup k' recursive) as [[params0 metadata0 handler] | ]
      eqn:Hfound.
    + simpl. apply Hclosure; rewrite <- ?Hscoped; simpl;
        [ reflexivity | reflexivity | exact Hfacts | exact Hconts ].
    + simpl. specialize (Hconts k').
      destruct (Pos.eqb k' k) eqn:Heq;
        [ | rewrite andb_false_r in Hconts; exact Hconts ].
      apply Pos.eqb_eq in Heq. rewrite Heq in Hfound.
      unfold is_bound in Hscoped. rewrite Hfound in Hscoped.
      simpl in Hscoped. rewrite andb_true_r in Hscoped, Hconts.
      rewrite <- Hscoped in Hconts. exact Hconts.
Qed.

(* A lock-step answer is an answer. *)
Lemma lockstep_answers (s s' : state) :
  result_simulated state_rel (step s) (step s') -> answers state_rel s s'.
Proof.
  unfold answers.
  destruct (step s) as [t | k0 vs hp0 | ] eqn:Hstep,
    (step s') as [t' | k0' vs' hp0' | ] eqn:Hstep'; simpl; try contradiction;
    try (intros _; left; exists 1; simpl; rewrite Hstep; reflexivity).
  - intros Hrel. right. right. left. exists 0, t, t'. simpl.
    rewrite Hstep. auto.
  - intros [<- [<- <-]]. right. left. exists k0, vs, hp0. auto.
Qed.

(* The same full call, returning and raising to related
   continuations. *)
Lemma call_code_rel (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) :
  cont_rel statics false return_to return_to' ->
  cont_rel statics false exn_to exn_to' ->
  result_simulated state_rel
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp)
    (call_code closure c vs args_arity return_arity return_to' exn_to'
       statics hp).
Proof.
  intros Hreturn Hraise.
  destruct (call_code_cases closure c vs args_arity return_arity statics hp)
    as [Hundefined | (callee_vars & Hcall)].
  - rewrite !Hundefined. exact I.
  - rewrite !Hcall. simpl.
    apply (State_rel false no_facts); [ reflexivity | apply no_facts_hold | ].
    intros k'. simpl.
    destruct (Pos.eq_dec k' (code_return_continuation c)) as [-> | Hne];
      [ rewrite !resolve_head; exact Hreturn | ].
    rewrite !(resolve_other k' (code_return_continuation c)) by exact Hne.
    destruct (Pos.eq_dec k' (code_exn_continuation c)) as [-> | Hne'];
      [ rewrite !resolve_head; exact Hraise | ].
    rewrite !(resolve_other k' (code_exn_continuation c)) by exact Hne'.
    apply Same_rel.
Qed.

(* The same call of any shape, where the two return continuations take
   a partial application to related results. *)
Lemma run_call_rel (shape : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) :
  cont_rel statics false return_to return_to' ->
  cont_rel statics false exn_to exn_to' ->
  (∀ partial : value,
    result_simulated state_rel
      (apply_cont_value return_to [partial] statics hp)
      (apply_cont_value return_to' [partial] statics hp)) ->
  result_simulated state_rel
    (run_call (fun partial => apply_cont_value return_to [partial] statics hp)
       shape return_arity return_to exn_to statics hp)
    (run_call (fun partial => apply_cont_value return_to' [partial] statics hp)
       shape return_arity return_to' exn_to' statics hp).
Proof.
  intros Hreturn Hraise Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply call_code_rel; assumption.
  - apply call_code_rel;
      [ apply Over_application_rel; assumption | exact Hraise ].
  - apply Hpartial.
  - exact I.
Qed.

(* Applying continuations that are related and are not the one being
   unboxed. *)
Lemma apply_cont_value_rel (statics : static_env) (hp : heap) (inside : bool)
    (c c' : cont_value) :
  cont_rel statics inside c c' ->
  inside = false ->
  ∀ vs : list value,
    result_simulated state_rel (apply_cont_value c vs statics hp)
      (apply_cont_value c' vs statics hp).
Proof.
  intros Hc.
  induction Hc
    as [ c | vars conts _
       | inside known params0 metadata0 handler handler' vars conts conts'
           recursive recursive' Hrecursive Hhandler Hfacts Hconts _
       | c c' extra Hc IH
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise _
       | known vars conts _ _ ];
    intros Hinside vs; try discriminate Hinside.
  - destruct (apply_cont_value c vs statics hp) as [t | k0 vs0 | ]; simpl;
      [ apply state_rel_refl | auto | exact I ].
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata0));
      [ | exact I ].
    destruct (bind_params params0 vs vars) as [bound | ] eqn:Hbind;
      [ | exact I ].
    simpl.
    apply (State_rel (inside && negb (is_bound k recursive))
      (facts_under_params known params0));
      [ exact Hhandler
      | exact (facts_hold_bind_params statics params0 known vs vars bound
          Hfacts Hbind)
      | exact (rec_conts_rel statics inside known vars conts conts' recursive
          recursive' Hrecursive Hfacts Hconts) ].
  - simpl. exact (IH eq_refl (vs ++ extra)).
  - rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; try exact I.
    unfold call_function.
    apply run_call_rel; [ exact Hreturn | exact Hraise | ].
    intros partial. exact (IHreturn eq_refl [partial]).
Qed.

Lemma apply_cont_rel (ac : apply_cont) (vars : env) (conts conts' : cont_env)
    (statics : static_env) (hp : heap) :
  cont_rel statics false (resolve (ac_continuation ac) conts)
    (resolve (ac_continuation ac) conts') ->
  result_simulated state_rel
    (step (State (Apply_cont ac) vars conts statics hp))
    (step (State (Apply_cont ac) vars conts' statics hp)).
Proof.
  intros Hc. rewrite !step_apply_cont.
  destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
  exact (apply_cont_value_rel statics _ false _ _ Hc eq_refl vs).
Qed.

(* The same call, returning and raising to related continuations. *)
Lemma apply_rel (ap : apply_expr) (vars : env) (conts conts' : cont_env)
    (statics : static_env) (hp : heap) :
  cont_rel statics false (resolve (ap_continuation ap) conts)
    (resolve (ap_continuation ap) conts') ->
  cont_rel statics false (resolve (ap_exn_continuation ap) conts)
    (resolve (ap_exn_continuation ap) conts') ->
  result_simulated state_rel (step (State (Apply ap) vars conts statics hp))
    (step (State (Apply ap) vars conts' statics hp)).
Proof.
  intros Hreturn Hraise. rewrite !step_apply.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | exact I ].
  unfold call_function.
  apply run_call_rel;
    [ exact Hreturn | apply with_extra_args_rel, Hraise | ].
  intros partial.
  exact (apply_cont_value_rel statics _ false _ _ Hreturn eq_refl [partial]).
Qed.

(* Extending the applications changes no static binding. *)
Fixpoint extend_binds_statics (e : expr) :
  ∀ (known : facts) (e' : expr),
    extend_applications_by extend known k e = Some e' ->
    binds_statics e' = binds_statics e
with extend_handler_binds_statics (h : continuation_handler) :
  ∀ (known : facts) (h' : continuation_handler),
    extend_in_handler_by extend known k h = Some h' ->
    handler_binds_statics h' = handler_binds_statics h.
Proof.
  - destruct e
      as [ y kind0 d body | bindings body | k0 h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known e'; simpl.
    + destruct
        (extend_applications_by extend (facts_under_let known y d) k body)
        as [body' | ] eqn:Hbody; [ | discriminate ].
      intros [= <-]. simpl. exact (extend_binds_statics body _ _ Hbody).
    + destruct
        (extend_applications_by extend (facts_under_statics known bindings) k
           body)
        as [body' | ] eqn:Hbody; [ | discriminate ].
      intros [= <-]. simpl.
      rewrite (extend_binds_statics body _ _ Hbody). reflexivity.
    + destruct (extend_in_handler_by extend known k h) as [h' | ] eqn:Hh;
        [ | discriminate ].
      destruct (Pos.eqb k0 k).
      * intros [= <-]. simpl.
        rewrite (extend_handler_binds_statics h _ _ Hh). reflexivity.
      * destruct (extend_applications_by extend known k body) as [body' | ]
          eqn:Hbody; [ | discriminate ].
        intros [= <-]. simpl.
        rewrite (extend_handler_binds_statics h _ _ Hh),
          (extend_binds_statics body _ _ Hbody).
        reflexivity.
    + destruct (is_bound k handlers); [ intros [= <-]; reflexivity | ].
      destruct
        (all_some
           (map
              (fun '(k', h) =>
                 let* h := extend_in_handler_by extend known k h in
                 Some (k', h))
              handlers))
        as [handlers' | ] eqn:Hhandlers; [ | discriminate ].
      destruct (extend_applications_by extend known k body) as [body' | ]
        eqn:Hbody; [ | discriminate ].
      intros [= <-]. simpl.
      rewrite (extend_binds_statics body _ _ Hbody). f_equal.
      clear Hbody. revert handlers' Hhandlers.
      induction handlers as [ | [k1 h1] handlers IH]; intros handlers'; simpl.
      * intros [= <-]. reflexivity.
      * destruct (extend_in_handler_by extend known k h1) as [h1' | ] eqn:Hh1;
          [ | discriminate ].
        destruct
          (all_some
             (map
                (fun '(k', h) =>
                   let* h := extend_in_handler_by extend known k h in
                   Some (k', h))
                handlers))
          as [rest_handlers | ]; [ | discriminate ].
        intros [= <-]. simpl.
        rewrite (extend_handler_binds_statics h1 _ _ Hh1),
          (IH rest_handlers eq_refl).
        reflexivity.
    + destruct
        (Pos.eqb (ap_continuation ap) k || Pos.eqb (ap_exn_continuation ap) k);
        [ discriminate | intros [= <-]; reflexivity ].
    + destruct (extend_apply_cont_by extend known k ac); [ | discriminate ].
      intros [= <-]. reflexivity.
    + destruct (extend_arms_by extend known k arms); [ | discriminate ].
      intros [= <-]. reflexivity.
    + intros [= <-]. reflexivity.
  - destruct h as [params0 metadata0 handler]. intros known h'. simpl.
    destruct
      (extend_applications_by extend (facts_under_params known params0) k
         handler)
      as [handler' | ] eqn:Hhandler; [ | discriminate ].
    intros [= <-]. simpl. exact (extend_binds_statics handler _ _ Hhandler).
Qed.

Lemma extended_handlers_bind_statics (known : facts)
    (handlers handlers' : list (continuation * continuation_handler)) :
  extended_handlers known handlers = Some handlers' ->
  handlers_bind_statics handlers' = handlers_bind_statics handlers.
Proof.
  unfold extended_handlers, handlers_bind_statics. revert handlers'.
  induction handlers as [ | [k1 h1] handlers IH]; intros handlers'; simpl.
  - intros [= <-]. reflexivity.
  - destruct (extend_in_handler_by extend known k h1) as [h1' | ] eqn:Hh1;
      [ | discriminate ].
    destruct
      (all_some
         (map
            (fun '(k', h) =>
               let* h := extend_in_handler_by extend known k h in
               Some (k', h))
            handlers))
      as [rest_handlers | ]; [ | discriminate ].
    intros [= <-]. simpl.
    rewrite (extend_handler_binds_statics h1 _ _ Hh1),
      (IH rest_handlers eq_refl).
    reflexivity.
Qed.

(* The arm found, extended. *)
Lemma find_arm_extended (known : facts) (j : Z) :
  ∀ arms arms' : list (Z * apply_cont),
    extend_arms_by extend known k arms = Some arms' ->
    match find_arm j arms with
    | Some ac =>
      ∃ ac' : apply_cont,
        extend_apply_cont_by extend known k ac = Some ac'
        ∧ find_arm j arms' = Some ac'
    | None => find_arm j arms' = None
    end.
Proof.
  induction arms as [ | [discriminant ac] arms IH]; intros arms'; simpl.
  - intros [= <-]. reflexivity.
  - destruct (extend_apply_cont_by extend known k ac) as [ac' | ] eqn:Hac;
      [ | discriminate ].
    destruct (extend_arms_by extend known k arms) as [rest_arms | ];
      [ | discriminate ].
    intros [= <-]. simpl. destruct (Z.eqb j discriminant).
    + exists ac'. auto.
    + exact (IH rest_arms eq_refl).
Qed.

(* The closure of a handler of a non-recursive [Let_cont] in the scope
   of the continuation. *)
Lemma closure_rel_in_scope (statics : static_env) (known : facts)
    (h h' : continuation_handler) (vars : env) (conts conts' : cont_env) :
  extend_in_handler_by extend known k h = Some h' ->
  facts_hold known vars statics ->
  conts_rel statics true conts conts' ->
  cont_rel statics false (Closure h vars conts []) (Closure h' vars conts' []).
Proof.
  destruct h as [params0 metadata0 handler]. simpl.
  destruct
    (extend_applications_by extend (facts_under_params known params0) k handler)
    as [handler' | ] eqn:Hhandler; [ | discriminate ].
  intros [= <-] Hfacts Hconts.
  apply (Closure_rel statics true known);
    [ reflexivity | exact Hhandler | exact Hfacts | exact Hconts ].
Qed.

Lemma closure_rel_out_of_scope (statics : static_env)
    (h : continuation_handler) (vars : env) (conts conts' : cont_env) :
  conts_rel statics false conts conts' ->
  cont_rel statics false (Closure h vars conts []) (Closure h vars conts' []).
Proof.
  destruct h as [params0 metadata0 handler]. intros Hconts.
  apply (Closure_rel statics false no_facts);
    [ reflexivity | reflexivity | apply no_facts_hold | exact Hconts ].
Qed.

(* ------------------------------------------------------------------------ *)
(* The simulation                                                           *)
(* ------------------------------------------------------------------------ *)

(* Out of the scope of the continuation the two runs step together. *)
Lemma out_of_scope_step (e : expr) (vars : env) (conts conts' : cont_env)
    (statics : static_env) (hp : heap) :
  conts_rel statics false conts conts' ->
  result_simulated state_rel (step (State e vars conts statics hp))
    (step (State e vars conts' statics hp)).
Proof.
  intros Hconts.
  destruct e
    as [ y kind0 d body | bindings body | k0 h body | handlers body | ap
       | ac | scrutinee arms | ].
  - rewrite !step_let.
    destruct (eval_named statics vars d) as [v | ]; simpl; [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind0); simpl; [ | exact I ].
    apply (State_rel false no_facts);
      [ reflexivity | apply no_facts_hold | exact Hconts ].
  - rewrite !step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings; simpl; [ | exact I ].
    apply (State_rel false no_facts);
      [ reflexivity | apply no_facts_hold | ].
    exact (conts_rel_extend statics extended false conts conts'
      (eval_static_bindings_extend vars bindings _ _ Hbindings) Hconts).
  - rewrite !step_let_cont_non_recursive. simpl.
    apply (State_rel false no_facts);
      [ reflexivity | apply no_facts_hold | ].
    exact (conts_rel_cons statics false k0 _ _ conts conts'
      (closure_rel_out_of_scope statics h vars conts conts' Hconts) Hconts).
  - rewrite !step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); simpl; [ exact I | ].
    apply (State_rel false no_facts);
      [ reflexivity | apply no_facts_hold | ].
    exact (rec_conts_rel statics false no_facts vars conts conts' handlers
      handlers eq_refl (no_facts_hold vars statics) Hconts).
  - exact (apply_rel ap vars conts conts' statics _
      (Hconts (ap_continuation ap)) (Hconts (ap_exn_continuation ap))).
  - exact (apply_cont_rel ac vars conts conts' statics _
      (Hconts (ac_continuation ac))).
  - rewrite !step_switch.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
    destruct (find_arm j arms) as [ac | ]; simpl; [ | exact I ].
    apply (State_rel false no_facts);
      [ reflexivity | apply no_facts_hold | exact Hconts ].
  - rewrite !step_invalid. exact I.
Qed.

(* An application of the continuation being unboxed: the first run
   enters the handler before; the second, with the new argument, the
   handler after. Where the continuation is not recursive the handler
   after refines the handler before. Where it is, the first run takes
   one step more, and the two are related again. *)
Lemma unboxed_application_answers (known : facts) (ac : apply_cont)
    (args' : list simple) (vars : env) (conts conts' : cont_env)
    (statics : static_env) (hp : heap) :
  ac_continuation ac = k ->
  extend known (ac_args ac) = Some args' ->
  facts_hold known vars statics ->
  conts_rel statics true conts conts' ->
  answers state_rel (State (Apply_cont ac) vars conts statics hp)
    (State
       (Apply_cont
          (Apply_cont_expr (ac_continuation ac) args' (ac_trap_action ac)))
       vars conts' statics hp).
Proof.
  intros Hk Hextend Hfacts Hconts.
  pose proof (Hconts k) as Hc. rewrite Pos.eqb_refl in Hc. simpl in Hc.
  inversion Hc
    as [ | def_vars def_conts Halike | | |
       | known0 def_vars def_conts Hdef_facts Henters ];
    match goal with
    | H : _ = resolve k conts |- _ => rename H into Hboxed
    end;
    match goal with
    | H : _ = resolve k conts' |- _ => rename H into Hunboxed
    end;
    (destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | left; exists 1; simpl; rewrite step_apply_cont, Hvs; reflexivity ]);
    (destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
       eqn:Hkinds;
      [ | left; exists 1; simpl; rewrite step_apply_cont, Hvs;
          unfold apply_continuation; rewrite Hk, <- Hboxed; simpl;
          rewrite Hkinds; reflexivity ]);
    (destruct (bind_params params vs def_vars) as [bound | ] eqn:Hbind;
      [ | left; exists 1; simpl; rewrite step_apply_cont, Hvs;
          unfold apply_continuation; rewrite Hk, <- Hboxed; simpl;
          rewrite Hkinds, Hbind; reflexivity ]);
    destruct (bind_params_value params vs def_vars bound i x Hbind x_at_i)
      as [v Hv];
    pose proof (bind_params_nth params vs def_vars bound i x v
      params_distinct Hbind x_at_i Hv) as Hx;
    destruct (extend_gives known vars statics hp (ac_args ac) args' vs v
                Hfacts Hextend Hvs Hv)
      as [(u & Hargs' & Hkind & Hfits) | Hwrong].
  - (* not recursive *)
    assert (Hfirst :
      step (State (Apply_cont ac) vars conts statics hp)
      = Step (State boxed bound def_conts statics hp)).
    { rewrite step_apply_cont, Hvs. unfold apply_continuation.
      rewrite Hk, <- Hboxed. simpl. rewrite Hkinds, Hbind. reflexivity. }
    right. right. right.
    eapply refines_step;
      [ exact Hfirst
      | | exact (Halike def_vars bound def_conts statics hp vs u Hbind
                   Hfits) ].
    rewrite step_apply_cont. simpl. rewrite Hargs'.
    unfold apply_continuation. rewrite Hk, <- Hunboxed. simpl.
    rewrite kinds_one_more,
      (kinds_snoc vs u _ kind Hkinds (proj2 (kind_eqb_eq _ _) Hkind)),
      (bind_params_snoc params f u vs def_vars bound Hbind).
    reflexivity.
  - assert (Hfirst :
      step (State (Apply_cont ac) vars conts statics hp)
      = Step (State boxed bound def_conts statics hp)).
    { rewrite step_apply_cont, Hvs. unfold apply_continuation.
      rewrite Hk, <- Hboxed. simpl. rewrite Hkinds, Hbind. reflexivity. }
    left. destruct (Hwrong bound def_conts Hx) as [n Hn]. exists (S n).
    rewrite run_succ, Hfirst. exact Hn.
  - (* recursive *)
    set (loop := (k, Handler params metadata boxed)) in *.
    set (loop' := (k, Handler (params ++ [f]) metadata' unboxed)) in *.
    assert (Hfirst :
      step (State (Apply_cont ac) vars conts statics hp)
      = Step
          (State boxed bound
             (rec_closures [loop] def_vars def_conts ++ def_conts) statics
             hp)).
    { rewrite step_apply_cont, Hvs. unfold apply_continuation.
      rewrite Hk, <- Hboxed. simpl. rewrite Hkinds, Hbind. reflexivity. }
    destruct
      (Henters def_vars bound
         (rec_closures [loop] def_vars def_conts ++ def_conts) statics hp vs
         u Hdef_facts Hbind Hfits)
      as [[n Hn] | (rest & inside & Hstep & Hrest & Hinside)].
    + left. exists (S n). rewrite run_succ, Hfirst. exact Hn.
    + right. right. left.
      exists 1,
        (State rest ((f, u) :: bound)
           (rec_closures [loop] def_vars def_conts ++ def_conts) statics hp),
        (State unboxed ((f, u) :: bound)
           (rec_closures [loop'] def_vars def_conts ++ def_conts) statics hp).
      split; [ | split ].
      * rewrite run_succ, Hfirst, run_succ, Hstep. reflexivity.
      * rewrite step_apply_cont. simpl. rewrite Hargs'.
        unfold apply_continuation. rewrite Hk, <- Hunboxed. simpl.
        rewrite kinds_one_more,
          (kinds_snoc vs u _ kind Hkinds (proj2 (kind_eqb_eq _ _) Hkind)),
          (bind_params_snoc params f u vs def_vars bound Hbind).
        reflexivity.
      * apply (State_rel true inside); [ exact Hrest | exact Hinside | ].
        intros k'. unfold rec_closures, loop, loop'. simpl.
        destruct (Pos.eqb_spec k' k) as [Heq | Hne].
        -- rewrite Heq, !resolve_head.
           apply (Unboxed_loop_rel statics known0); assumption.
        -- rewrite !resolve_other by exact Hne. apply Same_rel.
  - set (loop := (k, Handler params metadata boxed)) in *.
    assert (Hfirst :
      step (State (Apply_cont ac) vars conts statics hp)
      = Step
          (State boxed bound
             (rec_closures [loop] def_vars def_conts ++ def_conts) statics
             hp)).
    { rewrite step_apply_cont, Hvs. unfold apply_continuation.
      rewrite Hk, <- Hboxed. simpl. rewrite Hkinds, Hbind. reflexivity. }
    left.
    destruct
      (Hwrong bound (rec_closures [loop] def_vars def_conts ++ def_conts) Hx)
      as [n Hn].
    exists (S n). rewrite run_succ, Hfirst. exact Hn.
Qed.

Lemma state_rel_answers (s s' : state) :
  state_rel s s' -> answers state_rel s s'.
Proof.
  intros [inside known e e' vars conts conts' statics hp He Hfacts Hconts].
  destruct inside; unfold expr_rel in He;
    [ | subst e'; apply lockstep_answers, out_of_scope_step, Hconts ].
  destruct e
    as [ y kind0 d body | bindings body | k0 h body | handlers body | ap
       | ac | scrutinee arms | ];
    simpl in He.
  - (* Let *)
    destruct (extend_applications_by extend (facts_under_let known y d) k body)
      as [body' | ] eqn:Hbody; [ | discriminate ].
    injection He as <-. apply lockstep_answers. rewrite !step_let.
    destruct (eval_named statics vars d) as [v | ] eqn:Hd; simpl;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind0); simpl; [ | exact I ].
    apply (State_rel true (facts_under_let known y d));
      [ exact Hbody
      | exact (facts_hold_under_let known y d vars statics v Hfacts Hd)
      | exact Hconts ].
  - (* Let_static *)
    destruct
      (extend_applications_by extend (facts_under_statics known bindings) k
         body)
      as [body' | ] eqn:Hbody; [ | discriminate ].
    injection He as <-. apply lockstep_answers. rewrite !step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings; simpl; [ | exact I ].
    apply (State_rel true (facts_under_statics known bindings));
      [ exact Hbody
      | exact (facts_hold_under_statics known vars statics bindings extended
          Hfacts Hbindings)
      | exact (conts_rel_extend statics extended true conts conts'
          (eval_static_bindings_extend vars bindings _ _ Hbindings) Hconts) ].
  - (* Let_cont_non_recursive *)
    destruct (extend_in_handler_by extend known k h) as [h' | ] eqn:Hh;
      [ | discriminate ].
    pose proof (conts_rel_cons statics true k0 _ _ conts conts'
      (closure_rel_in_scope statics known h h' vars conts conts' Hh Hfacts
         Hconts)
      Hconts) as Hbound.
    destruct (Pos.eqb k0 k).
    + injection He as <-. apply lockstep_answers.
      rewrite !step_let_cont_non_recursive. simpl.
      apply (State_rel false no_facts);
        [ reflexivity | apply no_facts_hold | exact Hbound ].
    + destruct (extend_applications_by extend known k body) as [body' | ]
        eqn:Hbody; [ | discriminate ].
      injection He as <-. apply lockstep_answers.
      rewrite !step_let_cont_non_recursive. simpl.
      apply (State_rel true known);
        [ exact Hbody | exact Hfacts | exact Hbound ].
  - (* Let_cont_recursive *)
    pose proof (rec_conts_rel statics true known vars conts conts' handlers)
      as Hrec.
    destruct (is_bound k handlers).
    + injection He as <-. apply lockstep_answers.
      rewrite !step_let_cont_recursive.
      destruct (handlers_bind_statics handlers); simpl; [ exact I | ].
      apply (State_rel false no_facts);
        [ reflexivity | apply no_facts_hold | ].
      exact (Hrec handlers eq_refl Hfacts Hconts).
    + destruct
        (all_some
           (map
              (fun '(k', h) =>
                 let* h := extend_in_handler_by extend known k h in
                 Some (k', h))
              handlers))
        as [handlers' | ] eqn:Hhandlers; [ | discriminate ].
      destruct (extend_applications_by extend known k body) as [body' | ]
        eqn:Hbody; [ | discriminate ].
      injection He as <-. apply lockstep_answers.
      rewrite !step_let_cont_recursive.
      rewrite (extended_handlers_bind_statics known handlers handlers'
        Hhandlers).
      destruct (handlers_bind_statics handlers); simpl; [ exact I | ].
      apply (State_rel true known); [ exact Hbody | exact Hfacts | ].
      exact (Hrec handlers' Hhandlers Hfacts Hconts).
  - (* Apply *)
    destruct (Pos.eqb (ap_continuation ap) k) eqn:Hreturn;
      [ discriminate | ].
    destruct (Pos.eqb (ap_exn_continuation ap) k) eqn:Hraise;
      [ discriminate | ].
    injection He as <-. apply lockstep_answers.
    pose proof (Hconts (ap_continuation ap)) as Hc.
    pose proof (Hconts (ap_exn_continuation ap)) as Hc'.
    rewrite Hreturn in Hc. rewrite Hraise in Hc'.
    exact (apply_rel ap vars conts conts' statics _ Hc Hc').
  - (* Apply_cont *)
    destruct (extend_apply_cont_by extend known k ac) as [ac' | ] eqn:Hac;
      [ | discriminate ].
    injection He as <-. unfold extend_apply_cont_by in Hac.
    destruct (Pos.eqb (ac_continuation ac) k) eqn:Hk.
    + destruct (extend known (ac_args ac)) as [args' | ] eqn:Hextend;
        [ | discriminate ].
      injection Hac as <-. apply Pos.eqb_eq in Hk.
      exact (unboxed_application_answers known ac args' vars conts conts'
        statics _ Hk Hextend Hfacts Hconts).
    + injection Hac as <-. apply lockstep_answers.
      pose proof (Hconts (ac_continuation ac)) as Hc. rewrite Hk in Hc.
      exact (apply_cont_rel ac vars conts conts' statics _ Hc).
  - (* Switch *)
    destruct (extend_arms_by extend known k arms) as [arms' | ] eqn:Harms;
      [ | discriminate ].
    injection He as <-. apply lockstep_answers. rewrite !step_switch.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
    pose proof (find_arm_extended known j arms arms' Harms) as Hfind.
    destruct (find_arm j arms) as [ac | ].
    + destruct Hfind as (ac' & Hac & ->). simpl.
      apply (State_rel true known); [ | exact Hfacts | exact Hconts ].
      unfold expr_rel. simpl. rewrite Hac. reflexivity.
    + rewrite Hfind. exact I.
  - (* Invalid *)
    injection He as <-. apply lockstep_answers. rewrite !step_invalid.
    exact I.
Qed.

(* The handler given one more parameter, where every application in the
   body can be given its argument. *)
Lemma unbox_parameter_refines :
  handlers_alike ->
  ∀ (known : facts) (body body' : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    extend_applications_by extend known k body = Some body' ->
    facts_hold known vars statics ->
    refines
      (State (Let_cont_non_recursive k (Handler params metadata boxed) body)
         vars conts statics hp)
      (State
         (Let_cont_non_recursive k
            (Handler (params ++ [f]) metadata' unboxed) body')
         vars conts statics hp).
Proof.
  intros Halike known body body' vars conts statics hp Hbody Hfacts.
  apply let_cont_non_recursive_refines_step.
  apply (ahead_refines state_rel state_rel_answers).
  apply (State_rel true known); [ exact Hbody | exact Hfacts | ].
  intros k'. destruct (Pos.eqb_spec k' k) as [Heq | Hne].
  - rewrite Heq, !resolve_head. apply Unboxed_rel, Halike.
  - rewrite !resolve_other by exact Hne. apply Same_rel.
Qed.

(* The same for a recursive continuation that is alone in its
   [Let_cont]: the applications in the body are given their argument,
   and the handler after is the extension of what the handler before
   comes to in one step. *)
Lemma unbox_loop_parameter_refines (known : facts) (body body' : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap) :
  enters_loop known ->
  binds_statics unboxed = binds_statics boxed ->
  extend_applications_by extend known k body = Some body' ->
  facts_hold known vars statics ->
  refines
    (State (Let_cont_recursive [(k, Handler params metadata boxed)] body)
       vars conts statics hp)
    (State
       (Let_cont_recursive [(k, Handler (params ++ [f]) metadata' unboxed)]
          body')
       vars conts statics hp).
Proof.
  intros Henters Hbinds Hbody Hfacts.
  apply let_cont_recursive_refines_step;
    [ unfold handlers_bind_statics; simpl; rewrite Hbinds; reflexivity | ].
  intros _. apply (ahead_refines state_rel state_rel_answers).
  apply (State_rel true known); [ exact Hbody | exact Hfacts | ].
  intros k'. unfold rec_closures. simpl.
  destruct (Pos.eqb_spec k' k) as [Heq | Hne].
  - rewrite Heq, !resolve_head.
    apply (Unboxed_loop_rel statics known); assumption.
  - rewrite !resolve_other by exact Hne. apply Same_rel.
Qed.

End One_parameter.

(* ======================================================================== *)
(* A [Let] that reads a parameter, removed from where it is                 *)
(* ======================================================================== *)

(* What is asked of the value [u] given for the new parameter, the
   parameter read being [v]: whatever is bound later, the primitive has
   on [v] the value [u] if it has one. *)
Definition gives (p : unary_primitive) (v u : value) (statics : static_env)
    : Prop :=
  ∀ statics' : static_env,
    statics_extend statics statics' ->
    ∀ w : value, eval_prim statics' (Unary p) [v] = Some w -> w = u.

Lemma gives_extend (p : unary_primitive) (v u : value)
    (statics statics' : static_env) :
  gives p v u statics -> statics_extend statics statics' ->
  gives p v u statics'.
Proof.
  intros Hgives Hmore statics'' Hmore'.
  exact (Hgives statics'' (statics_extend_trans _ _ _ Hmore Hmore')).
Qed.

Section Read_removed.

(* The variable of the [Let], its kind, the primitive, and the
   parameter read. *)
Variable f : variable.
Variable kind : flambda_kind.
Variable p : unary_primitive.
Variable x : variable.

(* The second expression is the first without one [Let] of [f] to the
   primitive applied to [x], which is under [Let]s and in handlers or
   bodies of non-recursive [Let_cont]s that bind neither [x] nor
   [f]. *)
Inductive removes : expr -> expr -> Prop :=
  | Removes_here (rest : expr) :
    removes (Let f kind (Prim (Unary p) [Var x]) rest) rest
  | Removes_under_let (y : variable) (kind0 : flambda_kind) (d : named)
      (body body' : expr) :
    Pos.eqb x y = false ->
    Pos.eqb f y = false ->
    removes body body' ->
    removes (Let y kind0 d body) (Let y kind0 d body')
  | Removes_in_handler (k : continuation) (params : list variable)
      (metadata : handler_metadata) (handler handler' body : expr) :
    among x params = false ->
    among f params = false ->
    removes handler handler' ->
    removes (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body)
  | Removes_in_body (k : continuation) (h : continuation_handler)
      (body body' : expr) :
    removes body body' ->
    removes (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k h body').

(* Under variables that give [x] a value and [f] what the primitive has
   on it, the expression without the [Let] refines the expression. *)
Lemma removes_refines (v u : value) (e e' : expr) :
  removes e e' ->
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap),
    lookup x vars = Some v ->
    lookup f vars = Some u ->
    gives p v u statics ->
    refines (State e vars conts statics hp) (State e' vars conts statics hp).
Proof.
  intros Hremoves.
  induction Hremoves
    as [ rest | y kind0 d body body' Hx Hf _ IH
       | k params metadata handler handler' body Hx Hf _ IH
       | k h body body' _ IH ];
    intros vars conts statics hp Hlookup_x Hlookup_f Hgives.
  - destruct (eval_prim statics (Unary p) [v]) as [w | ] eqn:Hprim;
      [ | apply step_undefined_refines; rewrite step_let; simpl;
          rewrite Hlookup_x; simpl; rewrite Hprim; reflexivity ].
    pose proof (Hgives statics (statics_extend_refl statics) w Hprim) as ->.
    destruct (kind_eqb (kind_of_value u) kind) eqn:Hkind;
      [ | apply step_undefined_refines; rewrite step_let; simpl;
          rewrite Hlookup_x; simpl; rewrite Hprim, Hkind; reflexivity ].
    apply (refines_trans _ (State rest ((f, u) :: vars) conts statics hp)).
    + apply equivalent_refines, step_equivalent. rewrite step_let. simpl.
      rewrite Hlookup_x. simpl. rewrite Hprim, Hkind. reflexivity.
    + apply equivalent_refines, coincide_equivalent.
      * intros y _. destruct (Pos.eq_dec y f) as [-> | Hne];
          [ rewrite lookup_head; exact Hlookup_f
          | rewrite lookup_other by exact Hne; reflexivity ].
      * intros k _. apply conts_agree_same. reflexivity.
  - apply let_refines_step; [ reflexivity | ].
    intros v0 _ _. apply Pos.eqb_neq in Hx, Hf.
    apply IH; [ | | exact Hgives ]; rewrite lookup_other by assumption;
      assumption.
  - apply (let_cont_handler_refines_where (gives p v u) (gives_extend p v u));
      [ exact Hgives | ].
    intros vs handler_vars conts0 statics0 hp' Hgives0 _ Hbind.
    apply IH; [ | | exact Hgives0 ].
    + rewrite (bind_params_other params x vs vars handler_vars Hx Hbind).
      exact Hlookup_x.
    + rewrite (bind_params_other params f vs vars handler_vars Hf Hbind).
      exact Hlookup_f.
  - apply let_cont_non_recursive_refines_step.
    apply IH; assumption.
Qed.

(* The same where the variables of the expression do not yet bind [f],
   which is not free in it. *)
Lemma read_refines (v u : value) (e e' : expr) (bound : env)
    (conts : cont_env) (statics : static_env) (hp : heap) :
  removes e e' ->
  var_free_b f e = false ->
  Pos.eqb x f = false ->
  lookup x bound = Some v ->
  gives p v u statics ->
  refines (State e bound conts statics hp)
    (State e' ((f, u) :: bound) conts statics hp).
Proof.
  intros Hremoves Hfree Hne Hlookup Hgives.
  apply (refines_trans _ (State e ((f, u) :: bound) conts statics hp)).
  - apply equivalent_refines, coincide_equivalent.
    + intros y Hy. apply lookup_other. intros ->.
      apply var_free_b_sound in Hy. congruence.
    + intros k _. apply conts_agree_same. reflexivity.
  - apply (removes_refines v u e e' Hremoves);
      [ rewrite lookup_other by (apply Pos.eqb_neq; exact Hne); exact Hlookup
      | apply lookup_head | exact Hgives ].
Qed.

End Read_removed.

(* What is asked of the value given for the new parameter, among the
   values that an application gives: it is what the primitive gives of
   the one at the position of the parameter read. *)
Definition read_fits (i : nat) (p : unary_primitive) (statics : static_env)
    (vs : list value) (u : value) : Prop :=
  ∃ v : value, nth_error vs i = Some v ∧ gives p v u statics.

Lemma read_fits_refines (f : variable) (kind : flambda_kind)
    (p : unary_primitive) (x : variable) (i : nat) (params : list variable)
    (e e' : expr) (def_vars bound : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (vs : list value) (u : value) :
  removes f kind p x e e' ->
  var_free_b f e = false ->
  Pos.eqb x f = false ->
  distinct params = true ->
  nth_error params i = Some x ->
  bind_params params vs def_vars = Some bound ->
  read_fits i p statics vs u ->
  refines (State e bound conts statics hp)
    (State e' ((f, u) :: bound) conts statics hp).
Proof.
  intros Hremoves Hfree Hne Hdistinct Hx Hbind (v & Hv & Hgives).
  exact (read_refines f kind p x v u e e' bound conts statics hp Hremoves Hfree
    Hne (bind_params_nth params vs def_vars bound i x v Hdistinct Hbind Hx Hv)
    Hgives).
Qed.

(* ======================================================================== *)
(* The [Let] that a handler begins with                                     *)
(* ======================================================================== *)

(* [extend_applications] is [extend_applications_by] at
   [extend_args]. *)
Lemma extend_arms_by_args (i : nat) (p : unary_primitive) (known : facts)
    (k : continuation) (arms : list (Z * apply_cont)) :
  extend_arms_by (fun known => extend_args known i p) known k arms
  = extend_arms known k i p arms.
Proof.
  induction arms as [ | [discriminant ac] arms IH]; simpl;
    [ reflexivity | rewrite IH; reflexivity ].
Qed.

Fixpoint extend_applications_by_args (i : nat) (p : unary_primitive)
    (k : continuation) (e : expr) :
  ∀ known : facts,
    extend_applications_by (fun known => extend_args known i p) known k e
    = extend_applications known k i p e
with extend_in_handler_by_args (i : nat) (p : unary_primitive)
    (k : continuation) (h : continuation_handler) :
  ∀ known : facts,
    extend_in_handler_by (fun known => extend_args known i p) known k h
    = extend_in_handler known k i p h.
Proof.
  - destruct e
      as [ y kind0 d body | bindings body | k0 h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known; simpl; try reflexivity.
    + rewrite extend_applications_by_args. reflexivity.
    + rewrite extend_applications_by_args. reflexivity.
    + rewrite extend_in_handler_by_args, extend_applications_by_args.
      reflexivity.
    + rewrite extend_applications_by_args.
      replace
        (map
           (fun '(k', h) =>
              let* h :=
                extend_in_handler_by (fun known => extend_args known i p)
                  known k h
              in
              Some (k', h))
           handlers)
        with
          (map
             (fun '(k', h) =>
                let* h := extend_in_handler known k i p h in
                Some (k', h))
             handlers);
        [ reflexivity | ].
      induction handlers as [ | [k1 h1] handlers IH]; simpl;
        [ reflexivity | ].
      rewrite (extend_in_handler_by_args i p k h1 known), IH. reflexivity.
    + rewrite extend_arms_by_args. reflexivity.
  - destruct h as [params0 metadata0 handler]. intros known. simpl.
    rewrite extend_applications_by_args. reflexivity.
Qed.

(* The argument that [extend_args] gives: where the primitive has a
   value of the kind of the [Let] on the parameter read, that value;
   otherwise the [Let], the first thing in the handler, has no step. *)
Lemma extend_args_gives (i : nat) (p : unary_primitive) (f x : variable)
    (kind : flambda_kind) (rest : expr) (known : facts) (vars : env)
    (statics : static_env) (hp : heap) (args args' : list simple)
      (vs : list value)
    (v : value) :
  facts_hold known vars statics ->
  extend_args known i p args = Some args' ->
  eval_simples vars args = Some vs ->
  nth_error vs i = Some v ->
  (∃ u : value,
     eval_simples vars args' = Some (vs ++ [u])
     ∧ kind_of_value u = kind
     ∧ read_fits i p statics vs u)
  ∨ (∀ (bound : env) (conts : cont_env),
       lookup x bound = Some v ->
       ∃ n : nat,
         run n
           (State (Let f kind (Prim (Unary p) [Var x]) rest) bound conts
              statics hp)
         = Undefined).
Proof.
  intros Hfacts Hextend Hvs Hv. unfold extend_args in Hextend.
  destruct (nth_error args i) as [arg | ] eqn:Hnth; [ | discriminate ].
  destruct (projection_of known p arg) as [projection | ] eqn:Hprojection;
    [ | discriminate ].
  injection Hextend as <-.
  destruct (eval_simples_nth vars args vs i arg Hvs Hnth)
    as (v' & Hv' & Harg).
  rewrite Hv in Hv'. injection Hv' as <-.
  destruct (eval_prim statics (Unary p) [v]) as [w | ] eqn:Hprim;
    [ | right; intros bound conts Hx; exists 1; simpl; rewrite step_let;
        simpl; rewrite Hx; simpl; rewrite Hprim; reflexivity ].
  destruct (kind_eqb (kind_of_value w) kind) eqn:Hkind;
    [ | right; intros bound conts Hx; exists 1; simpl; rewrite step_let;
        simpl; rewrite Hx; simpl; rewrite Hprim, Hkind; reflexivity ].
  pose proof (projection_of_sound known vars statics p arg projection v w
    Hfacts Hprojection Harg Hprim) as Hw.
  left. exists w. split; [ | split ].
  - exact (eval_simples_snoc vars args projection w vs Hvs Hw).
  - apply kind_eqb_eq, Hkind.
  - exists v. split; [ exact Hv | ].
    intros statics' Hmore w' Hprim'.
    pose proof (projection_of_sound known vars statics' p arg projection v
      w' (facts_hold_extend known vars statics statics' Hfacts Hmore)
      Hprojection Harg Hprim') as Hw'.
    congruence.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The [Let]s that a handler begins with, one after the other. *)
Lemma unbox_heads_refines (handler : expr) :
  ∀ (known : facts) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap),
    facts_hold known vars statics ->
    refines
      (State (Let_cont_non_recursive k (Handler params metadata handler) body)
         vars conts statics hp)
      (State (unbox_heads known k params metadata handler body) vars conts
         statics hp).
Proof.
  induction handler
    as [ f kind d rest IH | bindings rest _ | k0 h rest _ | handlers rest _
       | ap | ac | scrutinee arms | ];
    intros known k params metadata body vars conts statics hp' Hfacts;
    try (simpl; apply refines_refl).
  destruct d
    as [ s | pr args | code slots | fslot decls slots | slot closure
       | pfrom slot closure | move_from move_to closure ];
    try (simpl; apply refines_refl).
  destruct pr as [p | b | v | hp | rp]; try (simpl; apply refines_refl).
  destruct args as [ | arg args]; try (simpl; apply refines_refl).
  destruct arg as [x | sym | c]; try (simpl; apply refines_refl).
  destruct args as [ | arg' args]; try (simpl; apply refines_refl).
  simpl.
  destruct (may_take params metadata f) eqn:Htake; [ | apply refines_refl ].
  destruct (position x params) as [i | ] eqn:Hposition;
    [ | apply refines_refl ].
  destruct (extend_applications known k i p body) as [body' | ] eqn:Hbody;
    [ | apply refines_refl ].
  unfold may_take in Htake.
  apply andb_true_iff in Htake as [Htake Hfresh].
  apply andb_true_iff in Htake as [Htake _].
  apply andb_true_iff in Htake as [_ Hdistinct].
  apply negb_true_iff in Hfresh.
  assert (Hne : Pos.eqb x f = false).
  { destruct (Pos.eqb_spec x f) as [Heq | _]; [ | reflexivity ].
    rewrite <- Heq, (position_among x params i Hposition) in Hfresh.
    discriminate. }
  assert (Hfree :
    var_free_b f (Let f kind (Prim (Unary p) [Var x]) rest) = false).
  { simpl. rewrite Pos.eqb_refl, Pos.eqb_sym, Hne. reflexivity. }
  eapply refines_trans; [ | apply IH; exact Hfacts ].
  rewrite <- extend_applications_by_args in Hbody.
  exact (unbox_parameter_refines k i params metadata
    (with_one_more metadata kind) f x kind
    (Let f kind (Prim (Unary p) [Var x]) rest) rest
    (fun known => extend_args known i p) (read_fits i p) Hdistinct
    (position_nth x params i Hposition) eq_refl
    (extend_args_gives i p f x kind rest)
    (fun def_vars bound conts0 statics0 _ vs u Hbind Hfits =>
       read_fits_refines f kind p x i params _ rest def_vars bound conts0
         statics0 _ vs u (Removes_here f kind p x rest) Hfree Hne Hdistinct
         (position_nth x params i Hposition) Hbind Hfits)
    known body body' vars conts statics _ Hbody Hfacts).
Qed.

(* The handler of each non-recursive [Let_cont] and its body, by the
   induction, with what is known there; then the [Let]s that the handler
   begins with. *)
Fixpoint unbox_params_knowing_refines (e : expr) :
  ∀ (known : facts) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap),
    facts_hold known vars statics ->
    refines (State e vars conts statics hp)
      (State (unbox_params_knowing known e) vars conts statics hp).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap
       | ac | scrutinee arms | ];
    intros known vars conts statics hp Hfacts; simpl.
  - (* Let *)
    apply let_refines_step; [ reflexivity | ].
    intros v Hd _. apply unbox_params_knowing_refines.
    exact (facts_hold_under_let known x d vars statics v Hfacts Hd).
  - (* Let_static *)
    apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings; [ | exact I ].
    apply unbox_params_knowing_refines.
    exact (facts_hold_under_statics known vars statics bindings extended
      Hfacts Hbindings).
  - (* Let_cont_non_recursive *)
    destruct h as [params metadata handler].
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata
               (unbox_params_knowing (facts_under_params known params)
                  handler))
            body)
         vars conts statics hp)).
    + apply (let_cont_handler_refines_where (facts_hold known vars)
        (facts_hold_extend known vars));
        [ exact Hfacts | ].
      intros vs handler_vars conts0 statics0 hp' Hfacts0 _ Hbind.
      apply unbox_params_knowing_refines.
      exact (facts_hold_bind_params statics0 params known vs vars
        handler_vars Hfacts0 Hbind).
    + apply (refines_trans _
        (State
           (Let_cont_non_recursive k
              (Handler params metadata
                 (unbox_params_knowing (facts_under_params known params)
                    handler))
              (unbox_params_knowing known body))
           vars conts statics hp)).
      * apply let_cont_non_recursive_refines_step.
        apply unbox_params_knowing_refines, Hfacts.
      * apply unbox_heads_refines, Hfacts.
  - (* Let_cont_recursive *)
    apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
    apply unbox_params_knowing_refines, Hfacts.
  - apply refines_refl.
  - apply refines_refl.
  - apply refines_refl.
  - apply refines_refl.
Qed.

Theorem unbox_known_params_refines : unbox_known_params_refines_statement.
Proof.
  intros e vars conts statics hp. apply unbox_params_knowing_refines.
  apply no_facts_hold.
Qed.
