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


(* Proofs of the statement of FlambdaUnboxLoopsSpec.v. Nothing here
   needs review.

   One read made a parameter ([unbox_loop_read_refines]) is the
   simulation of FlambdaUnboxParamsProofs.v for a recursive
   continuation
   (FlambdaUnboxParamsProofs.unbox_loop_parameter_refines). What it
   asks is that the handler before, entered with the values of an
   application, is refined, with no exit made later, by what the
   handler after extends, with the new parameter bound to an argument
   that is as asked. Here that is the handler without the read, and
   what is asked of the argument is that the read, run where the
   application is, gives it ([read_runs]). A primitive that reads a
   box and has run gives the same wherever the heap evolves to, so
   the handler without the [Let], entered with its variable bound to
   that value, refines the handler
   (FlambdaUnboxParamsProofs.read_refines); and it exits no later, as
   the read is under [Let]s only ([removed_exits_no_later]). Where the
   read, run at the application, gives nothing of the kind of the
   [Let], the handler goes wrong ([spine_read_never]): the term shows
   of the argument that it holds no location, so that the read is the
   same on every heap, or that a [Let] in scope has made the read,
   which then gives a value (FlambdaUnboxParamsProofs.projection_of_reads).
   The read is not taken to the head of the handler first: with a
   heap, a read that gives nothing where the handler is entered may
   give a value under the [Let]s above it. What is known after the
   read is what the applications in the handler were extended with.
   The pass is that, for the reads of each parameter in turn, and the
   traversal carries what is known as
   FlambdaUnboxVariantsProofs.unbox_variants_knowing_refines does. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaHeapSpec
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextProofs FlambdaContextMemoryProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaCompareSpec
  FlambdaCompareProofs FlambdaBlockLoadSpec FlambdaUnboxSpec
  FlambdaUnboxParamsSpec FlambdaUnboxParamsProofs FlambdaUnboxVariantsSpec
  FlambdaUnboxVariantsProofs FlambdaUnboxInScopeSpec
  FlambdaUnboxInScopeProofs FlambdaUnboxLoopsSpec.
Import ListNotations.

(* ======================================================================== *)
(* A read under [Let]s only                                                 *)
(* ======================================================================== *)

(* Removing a [Let] of a primitive changes no static binding. *)
Lemma removes_binds_statics (f : variable) (kind : flambda_kind)
    (p : unary_primitive) (x : variable) (e e' : expr) :
  removes f kind p x e e' -> binds_statics e' = binds_statics e.
Proof.
  intros Hremoves.
  induction Hremoves
    as [ rest | y kind0 d body body' _ _ _ IH
       | k params metadata handler handler' body _ _ _ IH
       | k h body body' _ IH ];
    simpl; try rewrite IH; reflexivity.
Qed.

(* One expression under variables that agree on its free names runs
   alike, step for step: an exit of one run is the exit of the other
   after as many steps. *)
Lemma coincide_exit (n : nat) (e : expr) (vars vars' : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
    (k : continuation) (vs : list value) (hp0 : heap) (rd0 : read_record) :
  (∀ x : variable, var_free_in_expr x e -> lookup x vars' = lookup x vars) ->
  run n (State e vars conts statics hp rd) = Exit k vs hp0 rd0 ->
  run n (State e vars' conts statics hp rd) = Exit k vs hp0 rd0.
Proof.
  intros Hvars Hexit.
  assert (Hmatch :
    state_matches (State e vars conts statics hp rd)
      (State e vars' conts statics hp rd)).
  { apply (State_matches (variables_map identity_renaming)
             (continuations_map identity_renaming)).
    - apply expr_eqb_refl; apply reflexive_map_nil.
    - intros x x' Hx Hmatch. apply variables_map_graph in Hmatch. subst x'.
      symmetry. apply Hvars, Hx.
    - intros k0 k' Hk Hmatch. apply continuations_map_graph in Hmatch.
      subst k'. apply conts_agree_same. reflexivity.
    - apply statics_match_same. }
  pose proof (bisimulation_run state_matches matched_step n _ _ Hmatch)
    as Hrun.
  rewrite Hexit in Hrun.
  destruct (run n (State e vars' conts statics hp rd)); simpl in Hrun;
    try contradiction.
  destruct Hrun as [<- [<- [<- <-]]]. reflexivity.
Qed.

(* Where the read is under [Let]s only and the variables bind the
   variable of its [Let] already to what the read gives, an exit of
   the expression is one of the expression without the [Let], no
   later: each [Let] above the read is a step of both runs, and the
   read is a step of the first only. *)
Lemma removed_exits_no_later (f : variable) (kind : flambda_kind)
    (p : unary_primitive) (x : variable) (v u : value) (e e' : expr) :
  removes f kind p x e e' ->
  on_spine f e = true ->
  ∀ (n : nat) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record) (k : continuation) (vs : list value)
    (hp0 : heap) (rd0 : read_record),
    lookup x vars = Some v ->
    lookup f vars = Some u ->
    gives p v u statics hp rd ->
    run n (State e vars conts statics hp rd) = Exit k vs hp0 rd0 ->
    ∃ m : nat,
      m <= n ∧ run m (State e' vars conts statics hp rd) = Exit k vs hp0 rd0.
Proof.
  intros Hremoves.
  induction Hremoves
    as [ rest | y kind0 d body body' Hx Hf _ IH
       | k1 params metadata handler handler' body _ _ _ _
       | k1 h body body' _ _ ];
    intros Hspine n vars conts statics hp rd k vs hp0 rd0 Hlookup_x
      Hlookup_f Hgives Hexit;
    simpl in Hspine; try discriminate Hspine.
  - destruct n as [ | n]; [ simpl in Hexit; discriminate Hexit | ].
    rewrite run_succ, step_let_exec, exec_named_prim in Hexit.
    cbn [eval_simples eval_simple] in Hexit. rewrite Hlookup_x in Hexit.
    cbv beta iota in Hexit.
    destruct (exec_prim statics hp (Unary p) [v]) as [[w hp'] | ] eqn:Hprim;
      cbv beta iota in Hexit; [ | discriminate Hexit ].
    pose proof (Hgives statics hp rd (statics_extend_refl statics)
      (memory_evolves_refl hp rd) w hp' Hprim) as ->.
    pose proof (exec_prim_heap statics hp hp' (Unary p) [v] u
      (fun q => ltac:(discriminate)) Hprim) as ->.
    destruct (kind_eqb (kind_of_value u) kind); cbv beta iota in Hexit;
      [ | discriminate Hexit ].
    exists n. split; [ lia | ].
    apply (coincide_exit n rest ((f, u) :: vars) vars); [ | exact Hexit ].
    intros y _. destruct (Pos.eq_dec y f) as [-> | Hne];
      [ rewrite lookup_head; exact Hlookup_f
      | rewrite lookup_other by exact Hne; reflexivity ].
  - rewrite Hf in Hspine. simpl in Hspine.
    destruct n as [ | n]; [ simpl in Hexit; discriminate Hexit | ].
    rewrite run_succ, step_let_exec in Hexit.
    destruct (exec_named statics hp rd vars d) as [[[w hp1] rd1] | ] eqn:Hd;
      cbv beta iota in Hexit; [ | discriminate Hexit ].
    destruct (kind_eqb (kind_of_value w) kind0) eqn:Hkind;
      cbv beta iota in Hexit; [ | discriminate Hexit ].
    apply Pos.eqb_neq in Hx, Hf.
    pose proof (IH Hspine n ((y, w) :: vars) conts statics hp1 rd1 k vs hp0
      rd0) as IH'.
    rewrite !lookup_other in IH' by assumption.
    destruct
      (IH' Hlookup_x Hlookup_f
         (gives_evolves p v u statics hp hp1 rd rd1 Hgives
            (exec_named_memory_evolves statics hp hp1 rd rd1 vars d w Hd))
         Hexit)
      as (m & Hm & Hexit').
    exists (S m). split; [ lia | ].
    rewrite run_succ, step_let_exec, Hd, Hkind. exact Hexit'.
Qed.

(* Where the read is under [Let]s only and, wherever the heap and the
   record evolve to, the primitive run on what the parameter read holds
   gives nothing of the kind of the [Let], the expression goes wrong:
   each [Let] above the read has a step or none, and the read has
   none. *)
Lemma spine_read_never (f : variable) (kind : flambda_kind)
    (p : unary_primitive) (x : variable) (v : value) (e e' : expr) :
  removes f kind p x e e' ->
  on_spine f e = true ->
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    lookup x vars = Some v ->
    (∀ (hp' : heap) (rd' : read_record),
      memory_evolves hp rd hp' rd' ->
      match exec_prim statics hp' (Unary p) [v] with
      | Some (w, _) => kind_eqb (kind_of_value w) kind = false
      | None => True
      end) ->
    behaves (State e vars conts statics hp rd) Goes_wrong.
Proof.
  intros Hremoves.
  induction Hremoves
    as [ rest | y kind0 d body body' Hx Hf _ IH
       | k params metadata handler handler' body _ _ _ _
       | k h body body' _ _ ];
    intros Hspine vars conts statics hp rd Hlookup Hwrong; simpl in Hspine;
    try discriminate.
  - apply step_undefined_wrong. rewrite step_let_exec, exec_named_prim.
    cbn [eval_simples eval_simple]. rewrite Hlookup. cbv beta iota.
    specialize (Hwrong hp rd (memory_evolves_refl hp rd)).
    destruct (exec_prim statics hp (Unary p) [v]) as [[w hp'] | ];
      cbv beta iota; [ rewrite Hwrong | ]; reflexivity.
  - rewrite Hf in Hspine. simpl in Hspine.
    apply (step_behaves _ Goes_wrong). rewrite step_let_exec.
    destruct (exec_named statics hp rd vars d) as [[[v0 hp1] rd1] | ] eqn:Hd;
      [ | reflexivity ].
    destruct (kind_eqb (kind_of_value v0) kind0); [ | reflexivity ].
    simpl. apply IH; [ exact Hspine | | ].
    + rewrite lookup_other by (apply Pos.eqb_neq; exact Hx). exact Hlookup.
    + intros hp' rd' Hevolves. apply (Hwrong hp' rd').
      exact (memory_evolves_trans hp hp1 hp' rd rd1 rd'
        (exec_named_memory_evolves statics hp hp1 rd rd1 vars d v0 Hd)
        Hevolves).
Qed.

(* ======================================================================== *)
(* One read of a loop made a parameter                                      *)
(* ======================================================================== *)

Lemma nth_error_among (x : variable) (xs : list variable) (i : nat) :
  nth_error xs i = Some x -> among x xs = true.
Proof. intros Hnth. apply among_In. exact (nth_error_In xs i Hnth). Qed.

(* What is asked of the value given for the new parameter of a loop,
   among the values that an application gives: the read of the one at
   the position of the parameter read, run where the application is,
   gives it and changes nothing. *)
Definition read_runs (i : nat) (p : unary_primitive) (statics : static_env)
    (hp : heap) (rd : read_record) (vs : list value) (u : value) : Prop :=
  reads_a_box p = true
  ∧ ∃ v : value,
      nth_error vs i = Some v
      ∧ exec_prim statics hp (Unary p) [v] = Some (u, hp).

(* A read that has run gives the same from then on. *)
Lemma runs_gives (p : unary_primitive) (v u : value) (statics : static_env)
    (hp : heap) (rd : read_record) :
  reads_a_box p = true ->
  exec_prim statics hp (Unary p) [v] = Some (u, hp) ->
  gives p v u statics hp rd.
Proof.
  intros Hbox Hruns statics' hp' rd' Hmore Hevolves w hp'' Hexec.
  rewrite (read_stays statics statics' hp hp' rd rd' p v u Hbox Hruns Hmore
    Hevolves) in Hexec.
  injection Hexec as <- _. reflexivity.
Qed.

(* The argument that [extend_args] gives, where the read is under
   [Let]s only in the handler: where the primitive, run on the
   parameter read at the application, gives a value of the kind of the
   [Let], that value; otherwise the handler goes wrong. The read then
   gives nothing of that kind where the handler comes to it: it gave a
   value at the application, which it gives again; or it gave none on
   an argument that holds no location, and gives none on any heap. *)
Lemma extend_args_gives_on_spine (i : nat) (p : unary_primitive)
    (f x : variable) (kind : flambda_kind) (handler without : expr)
    (known : facts) (vars : env) (statics : static_env) (hp : heap)
    (rd : read_record) (args args' : list simple) (vs : list value)
    (v : value) :
  removes f kind p x handler without ->
  on_spine f handler = true ->
  facts_hold known vars statics hp rd ->
  extend_args known i p args = Some args' ->
  eval_simples vars args = Some vs ->
  nth_error vs i = Some v ->
  (∃ u : value,
     eval_simples vars args' = Some (vs ++ [u])
     ∧ kind_of_value u = kind
     ∧ read_runs i p statics hp rd vs u)
  ∨ (∀ (bound : env) (conts : cont_env),
       lookup x bound = Some v ->
       ∃ n : nat,
         run n (State handler bound conts statics hp rd) = Undefined).
Proof.
  intros Hremoves Hspine Hfacts Hextend Hvs Hv. unfold extend_args in Hextend.
  destruct (nth_error args i) as [arg | ] eqn:Hnth; [ | discriminate ].
  destruct (projection_of known p arg) as [projection | ] eqn:Hprojection;
    [ | discriminate ].
  injection Hextend as <-.
  destruct (eval_simples_nth vars args vs i arg Hvs Hnth)
    as (v' & Hv' & Harg).
  rewrite Hv in Hv'. injection Hv' as <-.
  assert (Hbox : reads_a_box p = true).
  { unfold projection_of in Hprojection.
    destruct (reads_a_box p); [ reflexivity | discriminate ]. }
  assert (Hnever :
    (∀ (hp' : heap) (rd' : read_record),
      memory_evolves hp rd hp' rd' ->
      match exec_prim statics hp' (Unary p) [v] with
      | Some (w, _) => kind_eqb (kind_of_value w) kind = false
      | None => True
      end) ->
    ∀ (bound : env) (conts : cont_env),
      lookup x bound = Some v ->
      ∃ n : nat,
        run n (State handler bound conts statics hp rd) = Undefined).
  { intros Hwrong bound conts Hx.
    exact (spine_read_never f kind p x v handler without Hremoves Hspine
      bound conts statics hp rd Hx Hwrong). }
  destruct (exec_prim statics hp (Unary p) [v]) as [[w hp'] | ] eqn:Hprim.
  - pose proof (exec_prim_heap statics hp hp' (Unary p) [v] w
      (fun q => ltac:(discriminate)) Hprim) as ->.
    destruct (kind_eqb (kind_of_value w) kind) eqn:Hkind.
    + left. exists w. split; [ | split ].
      * exact (eval_simples_snoc vars args projection w vs Hvs
          (projection_of_sound known vars statics hp hp rd p arg projection
             v w Hfacts Hprojection Harg Hprim)).
      * apply kind_eqb_eq, Hkind.
      * split; [ exact Hbox | ]. exists v. split; [ exact Hv | exact Hprim ].
    + right. apply Hnever. intros hp2 rd2 Hevolves.
      rewrite (read_stays statics statics hp hp2 rd rd2 p v w Hbox Hprim
        (statics_extend_refl statics) Hevolves).
      exact Hkind.
  - right. apply Hnever. intros hp2 rd2 Hevolves.
    destruct (projection_of_reads known vars statics hp rd p arg projection
                v Hfacts Hprojection Harg)
      as [Hpure | [w Hruns]];
      [ | rewrite Hruns in Hprim; discriminate Hprim ].
    rewrite (exec_prim_no_location statics hp2 (Unary p) [v]);
      [ | intros q; discriminate | intros l [Heq | []]; exact (Hpure l Heq) ].
    rewrite (exec_prim_no_location statics hp (Unary p) [v]) in Hprim;
      [ | intros q; discriminate | intros l [Heq | []]; exact (Hpure l Heq) ].
    destruct (eval_prim statics (Unary p) [v]); simpl in Hprim |- *;
      [ discriminate Hprim | exact I ].
Qed.

Lemma unbox_loop_read_refines (known : facts) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) (r : read) (x : variable) (state : loop_state)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  nth_error params (read_position r) = Some x ->
  removes (read_variable r) (read_kind r) (read_primitive r) x handler
    (read_without r) ->
  unbox_loop_read known k params metadata handler body r = Some state ->
  facts_hold known vars statics hp rd ->
  refines (State (loop_of k (params, metadata, handler, body)) vars conts
             statics hp rd)
    (State (loop_of k state) vars conts statics hp rd).
Proof.
  intros Hx Hremoves Hread Hfacts. unfold unbox_loop_read in Hread.
  rewrite Hx in Hread. simpl in Hread.
  destruct r as [f kind p i without]. simpl in *.
  destruct
    (may_take params metadata f && negb (var_free_b f handler)
     && on_spine f handler)
    eqn:Hchecks; [ | discriminate ].
  apply andb_true_iff in Hchecks as [Hchecks Hspine].
  apply andb_true_iff in Hchecks as [Htake Hfree].
  apply negb_true_iff in Hfree.
  destruct (extend_applications known k i p body) as [body' | ] eqn:Hbody;
    [ | discriminate ].
  simpl in Hread.
  destruct
    (extend_applications
       (facts_under_let (facts_under_params known params) f
          (Prim (Unary p) [Var x]))
       k i p without)
    as [without' | ] eqn:Hwithout; [ | discriminate ].
  simpl in Hread. injection Hread as <-.
  unfold may_take in Htake.
  apply andb_true_iff in Htake as [Htake Hfresh].
  apply andb_true_iff in Htake as [Htake _].
  apply andb_true_iff in Htake as [_ Hdistinct].
  apply negb_true_iff in Hfresh.
  assert (Hne : Pos.eqb x f = false).
  { destruct (Pos.eqb_spec x f) as [Heq | _]; [ | reflexivity ].
    rewrite <- Heq, (nth_error_among x params i Hx) in Hfresh.
    discriminate. }
  unfold loop_of.
  rewrite <- extend_applications_by_args in Hbody, Hwithout.
  apply (unbox_loop_parameter_refines k i params metadata
    (with_one_more metadata kind) f x kind handler without'
    (fun known => extend_args known i p) (read_runs i p) Hdistinct Hx
    eq_refl
    (fun known0 vars0 statics0 hp0 rd0 args0 args0' vs0 v0 =>
       extend_args_gives_on_spine i p f x kind handler without known0 vars0
         statics0 hp0 rd0 args0 args0' vs0 v0 Hremoves Hspine)
    known body body' vars conts statics hp rd);
    [ | | exact Hbody | exact Hfacts ].
  - intros def_vars bound conts0 statics0 hp0 rd0 vs u Hfacts0 Hbind
      (Hbox & v & Hv & Hruns).
    pose proof (bind_params_nth params vs def_vars bound i x v Hdistinct
      Hbind Hx Hv) as Hlookup.
    pose proof (runs_gives p v u statics0 hp0 rd0 Hbox Hruns) as Hgives.
    right.
    exists without,
      (facts_under_let (facts_under_params known params) f
         (Prim (Unary p) [Var x])).
    split; [ split | split; [ exact Hwithout | ] ].
    + exact (read_refines f kind p x v u handler without bound conts0
        statics0 hp0 rd0 Hremoves Hfree Hne Hlookup Hgives).
    + intros n k0 vs0 hp1 rd1 Hexit.
      apply (removed_exits_no_later f kind p x v u handler without Hremoves
        Hspine n ((f, u) :: bound) conts0 statics0 hp0 rd0 k0 vs0 hp1 rd1);
        [ rewrite lookup_other by (apply Pos.eqb_neq; exact Hne);
          exact Hlookup
        | apply lookup_head | exact Hgives | ].
      apply (coincide_exit n handler bound ((f, u) :: bound));
        [ | exact Hexit ].
      intros y Hy. apply lookup_other. intros ->.
      apply var_free_b_sound in Hy. congruence.
    + apply (facts_hold_under_let (facts_under_params known params) f
        (Prim (Unary p) [Var x]) bound statics0 hp0 hp0 rd0 rd0 u);
        [ exact (facts_hold_bind_params statics0 hp0 rd0 params known vs
            def_vars bound Hfacts0 Hbind)
        | ].
      rewrite exec_named_prim. cbn [eval_simples eval_simple].
      rewrite Hlookup. cbv beta iota. rewrite Hruns. reflexivity.
  - rewrite (extend_binds_statics k (fun known => extend_args known i p)
      without _ without' Hwithout).
    exact (removes_binds_statics f kind p x handler without Hremoves).
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

Lemma unbox_loop_param_refines (fuel : nat) :
  ∀ (known : facts) (k : continuation) (i : nat) (state : loop_state)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record),
    facts_hold known vars statics hp rd ->
    refines (State (loop_of k state) vars conts statics hp rd)
      (State (loop_of k (unbox_loop_param fuel known k i state)) vars conts
         statics hp rd).
Proof.
  induction fuel as [ | fuel IH];
    intros known k i state vars conts statics hp rd Hfacts;
    [ apply refines_refl | ].
  destruct state as [[[params metadata] handler] body]. simpl.
  destruct (nth_error params i) as [param | ] eqn:Hparam; simpl;
    [ | apply refines_refl ].
  destruct (next_read [param] handler) as [r | ] eqn:Hnext; simpl;
    [ | apply refines_refl ].
  destruct
    (unbox_loop_read known k params metadata handler body
       (Read (read_variable r) (read_kind r) (read_primitive r) i
          (read_without r)))
    as [state' | ] eqn:Hread; [ | apply refines_refl ].
  eapply refines_trans; [ | apply IH; exact Hfacts ].
  destruct (next_read_first [param] handler r Hnext) as [wanted Hfirst].
  destruct (first_read_removes handler wanted [param] [] r Hfirst)
    as (x & Hposition & _ & _ & Hremoves).
  apply position_single in Hposition. subst x.
  exact (unbox_loop_read_refines known k params metadata handler body
    (Read (read_variable r) (read_kind r) (read_primitive r) i
       (read_without r))
    param state' vars conts statics hp rd Hparam Hremoves Hread Hfacts).
Qed.

Lemma unbox_loop_param_all_refines (fuel : nat) (known : facts)
    (k : continuation) (i : nat) (state : loop_state) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  facts_hold known vars statics hp rd ->
  refines (State (loop_of k state) vars conts statics hp rd)
    (State (loop_of k (unbox_loop_param_all fuel known k i state)) vars conts
       statics hp rd).
Proof.
  intros Hfacts. unfold unbox_loop_param_all.
  destruct (reads_left i (unbox_loop_param fuel known k i state));
    [ apply refines_refl | apply unbox_loop_param_refines, Hfacts ].
Qed.

Lemma unbox_loop_params_refines (fuel : nat) (known : facts)
    (k : continuation) (positions : list nat) :
  ∀ (state : loop_state) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    facts_hold known vars statics hp rd ->
    refines (State (loop_of k state) vars conts statics hp rd)
      (State (loop_of k (unbox_loop_params fuel known k positions state))
         vars conts statics hp rd).
Proof.
  induction positions as [ | i positions IH];
    intros state vars conts statics hp rd Hfacts; simpl;
    [ apply refines_refl | ].
  eapply refines_trans; [ | apply IH; exact Hfacts ].
  apply unbox_loop_param_all_refines, Hfacts.
Qed.

Lemma unbox_loop_reads_refines (known : facts) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  facts_hold known vars statics hp rd ->
  refines
    (State (Let_cont_recursive [(k, Handler params metadata handler)] body)
       vars conts statics hp rd)
    (State (unbox_loop_reads known k params metadata handler body) vars conts
       statics hp rd).
Proof.
  intros Hfacts. unfold unbox_loop_reads.
  exact (unbox_loop_params_refines _ known k _ (params, metadata, handler, body)
    vars conts statics hp rd Hfacts).
Qed.

Fixpoint unbox_loops_knowing_refines (e : expr) :
  ∀ (known : facts) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record),
    facts_hold known vars statics hp rd ->
    refines (State e vars conts statics hp rd)
      (State (unbox_loops_knowing known e) vars conts statics hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap
       | ac | scrutinee arms | ];
    intros known vars conts statics hp rd Hfacts; simpl.
  - apply let_refines_step_exec; [ reflexivity | ].
    intros v hp' rd' Hd _. apply unbox_loops_knowing_refines.
    exact (facts_hold_under_let known x d vars statics hp hp' rd rd' v
      Hfacts Hd).
  - apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings; [ | exact I ].
    apply unbox_loops_knowing_refines.
    exact (facts_hold_under_statics known vars statics bindings extended hp
      rd Hfacts Hbindings).
  - destruct h as [params metadata handler].
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata
               (unbox_loops_knowing (facts_under_params known params)
                  handler))
            body)
         vars conts statics hp rd)).
    + apply (let_cont_handler_refines_where_memory (facts_hold known vars)
        (facts_hold_extend known vars) (facts_hold_evolves known vars));
        [ exact Hfacts | ].
      intros vs handler_vars conts0 statics0 hp0 rd0 Hfacts0 _ Hbind.
      apply unbox_loops_knowing_refines.
      exact (facts_hold_bind_params statics0 hp0 rd0 params known vs vars
        handler_vars Hfacts0 Hbind).
    + apply let_cont_non_recursive_refines_step.
      apply unbox_loops_knowing_refines, Hfacts.
  - assert (Hbody :
      refines
        (State (Let_cont_recursive handlers body) vars conts statics hp rd)
        (State (Let_cont_recursive handlers (unbox_loops_knowing known body))
           vars conts statics hp rd)).
    { apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply unbox_loops_knowing_refines, Hfacts. }
    destruct handlers as [ | [k [params metadata handler]] [ | h2 handlers]];
      try exact Hbody.
    eapply refines_trans; [ exact Hbody | ].
    apply unbox_loop_reads_refines, Hfacts.
  - apply refines_refl.
  - apply refines_refl.
  - apply refines_refl.
  - apply refines_refl.
Qed.

Theorem unbox_loops_knowing_symbols_refines :
  unbox_loops_knowing_symbols_refines_statement.
Proof.
  intros known e vars conts statics hp rd Honly Hbound.
  apply unbox_loops_knowing_refines.
  exact (symbols_facts_hold known vars statics hp rd Honly Hbound).
Qed.
