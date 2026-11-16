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

(* Proofs of the statements of FlambdaLoopifySpec.v. Nothing here needs
   review.

   The term and the term with code made loops differ in what the static
   environment binds code ids to, and nowhere else: the two runs execute
   the same expressions until code that was made a loop is called. So
   the proof is first about two states with one expression under static
   environments that differ so ([statics_loop]), and then about the
   [Let_static] that makes them differ.

   The two runs are followed side by side, each step of the first
   answered by one or more of the second
   (FlambdaSimulationProofs.stuttering_refines). The relation
   ([state_loops]) has two cases.

   - Outside a loop the expressions are the same, the variables are the
     same, and each continuation name stands in the two states for
     related continuations ([cont_rel]).
   - Inside the body of code that was made a loop, since a call of it
     ([frame]: the code, the closure it was called with, and what the
     second run had at the entry), the second expression is the first
     with its self tail calls made jumps; the variables give the same
     value to every name, though the second has the parameters bound
     once more for each time round; [my_closure] is still the closure
     called; and in the second state [self] stands for the closure of
     the loop made at the entry, and the return and exception
     continuations for what they stood for there.

   A call of code that was made a loop takes one step in the first run
   and three in the second, which binds [self] and jumps to it; the two
   are then inside the loop. A self tail call there takes one step in
   each: the first calls the code again, which is found by the closure
   as it was at the entry ([frame_statics]), and the second jumps; the
   two are inside the loop again, with the same frame. A self tail call
   is a full call, so it is not a partial application nor an
   over-application.

   A call is of the same shape in the two runs
   (FlambdaSemanticsProofs.shape_of_call_related), of code that is the
   same or was made a loop. What is left of an over-application is a
   continuation that makes a call when it is applied, so applying
   related continuations may also take three steps in the second run
   ([result_loops]).

   The handler of [self] has no step unless the values it is applied to
   are of the kinds of the parameters of the code
   (FlambdaSemanticsSpec.apply_cont_value). They are: a call that has a
   step gave values of those kinds ([agreed_kinds]); at the entry the
   jump passes the parameters, which are distinct, so their values are
   the arguments of the call ([eval_params_bound]); and a self tail call
   is such a call too.

   A continuation closure made inside a loop may be applied much later,
   when the static environment has grown. What is known of the static
   environment where it was made still holds ([cont_rel_extend]); that
   is why the relation on continuations is indexed by the static
   environment of the first state. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaContextSpec FlambdaContextProofs FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaSimulationProofs FlambdaSinkContsProofs
  FlambdaUnusedContsSpec FlambdaUnusedParamsProofs FlambdaLoopifySpec.
Import ListNotations.

(* ======================================================================== *)
(* The body of the loop                                                     *)
(* ======================================================================== *)

Fixpoint loop_calls_binds_statics (my_closure : variable)
    (params_arity : arity) (ret exn self : continuation) (e : expr) :
  binds_statics (loop_calls my_closure params_arity ret exn self e)
  = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - apply loop_calls_binds_statics.
  - rewrite loop_calls_binds_statics. reflexivity.
  - rewrite !loop_calls_binds_statics. reflexivity.
  - rewrite loop_calls_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
      [ reflexivity | ].
    rewrite loop_calls_binds_statics, IH. reflexivity.
  - destruct (is_self_tail_call my_closure params_arity ret exn ap);
      reflexivity.
Qed.

Lemma loop_handlers_bind_statics (my_closure : variable)
    (params_arity : arity) (ret exn self : continuation)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map
       (fun '(k, h) =>
          (k, loop_calls_in_handler my_closure params_arity ret exn self h))
       handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite loop_calls_binds_statics, IH. reflexivity.
Qed.

Lemma lookup_loop_handlers (my_closure : variable) (params_arity : arity)
    (ret exn self k : continuation)
    (handlers : list (continuation * continuation_handler)) :
  lookup k
    (map
       (fun '(k, h) =>
          (k, loop_calls_in_handler my_closure params_arity ret exn self h))
       handlers)
  = option_map (loop_calls_in_handler my_closure params_arity ret exn self)
      (lookup k handlers).
Proof.
  induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

(* ======================================================================== *)
(* Binding parameters                                                       *)
(* ======================================================================== *)

(* Binding the same parameters to the same values under environments
   that agree outside the parameters gives environments that agree. *)
Lemma bind_params_agree_outside (params : list variable) (vs : list value)
    (vars vars' bound bound' : env) :
  bind_params params vs vars = Some bound ->
  bind_params params vs vars' = Some bound' ->
  (∀ x : variable, x ∉ params -> lookup x vars = lookup x vars') ->
  ∀ x : variable, lookup x bound = lookup x bound'.
Proof.
  intros Hbind Hbind' Houtside x. symmetry.
  apply (bind_params_agree params vs vars vars' bound bound' Hbind Hbind').
  destruct (in_dec Pos.eq_dec x params) as [Hin | Hnotin];
    [ left; exact Hin | right; symmetry; apply Houtside, Hnotin ].
Qed.

Lemma bind_params_ext (params : list variable) (vs : list value)
    (vars vars' bound : env) :
  (∀ x : variable, lookup x vars = lookup x vars') ->
  bind_params params vs vars = Some bound ->
  ∃ bound' : env,
    bind_params params vs vars' = Some bound'
    ∧ ∀ x : variable, lookup x bound = lookup x bound'.
Proof.
  intros Hext Hbind.
  destruct (bind_params_defined params vs vars vars' bound Hbind)
    as [bound' Hbind'].
  exists bound'. split; [ exact Hbind' | ].
  apply (bind_params_agree_outside params vs vars vars' bound bound' Hbind
    Hbind').
  intros x _. apply Hext.
Qed.

(* Parameters that are distinct, once bound, have the values they were
   bound to. *)
Lemma eval_params_bound (params : list variable) :
  ∀ (vs : list value) (base entry : env),
    distinct params = true ->
    bind_params params vs base = Some entry ->
    eval_simples entry (map Var params) = Some vs.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] base entry;
    simpl; try discriminate; [ reflexivity | ].
  intros Hdistinct Hbind. apply andb_true_iff in Hdistinct as [Hp Hdistinct].
  apply negb_true_iff in Hp.
  rewrite (lookup_bind_params_other params vs _ entry p Hbind),
    (IH vs _ entry Hdistinct Hbind).
  - simpl. rewrite Pos.eqb_refl. reflexivity.
  - intros Hin. apply among_In in Hin. congruence.
Qed.

(* A call that agrees with the code gives values of the kinds of its
   parameters. *)
Lemma agreed_kinds (vs : list value) (args_arity return_arity : arity)
    (c : code) :
  negb
    (kinds_eqb (map kind_of_value vs) (unarize args_arity)
     && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
          (unarize_per_parameter (code_params_arity c))
     && arity_eqb return_arity (code_result_arity c))
  = false ->
  map kind_of_value vs = code_params_kinds c.
Proof.
  intros Hagrees. apply negb_false_iff in Hagrees.
  apply andb_true_iff in Hagrees as [Hagrees _].
  apply andb_true_iff in Hagrees as [Hkinds Hargs].
  apply kinds_eqb_eq in Hkinds. apply kinds_per_parameter_eqb_eq in Hargs.
  unfold code_params_kinds.
  rewrite <- (unarize_per_parameter_eq _ _ Hargs). exact Hkinds.
Qed.

(* The parameters that are bound, evaluated and bound again to their own
   values, give an environment that agrees with the first. *)
Lemma rebind_params_aux (params : list variable) (vars : env) :
  (∀ p : variable, p ∈ params -> ∃ v : value, lookup p vars = Some v) ->
  ∃ vs : list value,
    eval_simples vars (map Var params) = Some vs
    ∧ ∀ base : env,
        (∀ x : variable, lookup x base = lookup x vars) ->
        ∃ bound : env,
          bind_params params vs base = Some bound
          ∧ ∀ x : variable, lookup x bound = lookup x vars.
Proof.
  induction params as [ | p params IH]; intros Hbound; simpl.
  - exists []. split; [ reflexivity | ].
    intros base Hbase. exists base. split; [ reflexivity | exact Hbase ].
  - destruct (Hbound p (or_introl eq_refl)) as [v Hv].
    destruct (IH (fun q Hin => Hbound q (or_intror Hin))) as (vs & Hvs & Hrest).
    exists (v :: vs). rewrite Hv, Hvs. split; [ reflexivity | ].
    intros base Hbase. apply Hrest. intros x. simpl.
    destruct (Pos.eqb_spec x p) as [-> | _];
      [ symmetry; exact Hv | apply Hbase ].
Qed.

Lemma rebind_params (params : list variable) (vs : list value)
    (base entry : env) :
  bind_params params vs base = Some entry ->
  ∃ (vs' : list value) (entry' : env),
    eval_simples entry (map Var params) = Some vs'
    ∧ bind_params params vs' entry = Some entry'
    ∧ ∀ x : variable, lookup x entry = lookup x entry'.
Proof.
  intros Hbind.
  destruct (rebind_params_aux params entry) as (vs' & Hvs' & Hrebind).
  { intros p Hin. apply (bind_params_bound params vs base entry p Hbind).
    left. exact Hin. }
  destruct (Hrebind entry (fun x => eq_refl)) as (entry' & Hbind' & Hsame).
  exists vs', entry'. split; [ exact Hvs' | ]. split; [ exact Hbind' | ].
  intros x. symmetry. apply Hsame.
Qed.

(* ======================================================================== *)
(* Static environments that differ in code made loops                       *)
(* ======================================================================== *)

(* The second code is the first, or the first made a loop. *)
Definition code_loops (c c' : code) : Prop :=
  c' = c
  ∨ ∃ self : continuation, loopable self c = true ∧ c' = looped_code self c.

Definition statics_loop (statics statics' : static_env) : Prop :=
  Forall2 (fun ic ic' => fst ic = fst ic' ∧ code_loops (snd ic) (snd ic'))
    (static_code statics) (static_code statics')
  ∧ static_values statics = static_values statics'.

Lemma statics_loop_refl (statics : static_env) : statics_loop statics statics.
Proof.
  split; [ | reflexivity ].
  induction (static_code statics) as [ | ic code IH];
    [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact IH ].
  split; [ reflexivity | left; reflexivity ].
Qed.

Lemma lookup_code_loop (id : code_id) (statics statics' : static_env) :
  statics_loop statics statics' ->
  match lookup id (static_code statics), lookup id (static_code statics') with
  | Some c, Some c' => code_loops c c'
  | None, None => True
  | _, _ => False
  end.
Proof.
  intros [Hcode _].
  induction Hcode as [ | [id1 c] [id1' c'] code code' [Hid Hc] _ IH]; simpl;
    [ exact I | ].
  simpl in Hid. subst id1'. destruct (Pos.eqb id id1); [ exact Hc | exact IH ].
Qed.

(* What the machine reads of the two is the same, but for the bodies of
   code. *)
Definition any_body (body body' : expr) : Prop := True.

Lemma looped_code_binds_statics (self : continuation) (c : code) :
  binds_statics (code_body (looped_code self c))
  = binds_statics (code_body c).
Proof.
  destruct c as [params my_closure ret exn metadata body]. simpl.
  rewrite loop_calls_binds_statics, !orb_false_r. reflexivity.
Qed.

(* The looped code is the code, but for its body. *)
Lemma looped_code_fields (self : continuation) (c : code) :
  code_params (looped_code self c) = code_params c
  ∧ code_my_closure (looped_code self c) = code_my_closure c
  ∧ code_return_continuation (looped_code self c)
    = code_return_continuation c
  ∧ code_exn_continuation (looped_code self c) = code_exn_continuation c
  ∧ code_params_arity (looped_code self c) = code_params_arity c
  ∧ code_result_arity (looped_code self c) = code_result_arity c.
Proof. destruct c. repeat split. Qed.

Lemma statics_loop_related (statics statics' : static_env) :
  statics_loop statics statics' ->
  statics_related any_body statics statics'.
Proof.
  intros [Hcode Hvalues]. split; [ | exact Hvalues ].
  induction Hcode as [ | [id c] [id' c'] code code' [Hid Hc] _ IH];
    [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact IH ].
  split; [ exact Hid | ]. simpl in Hc |- *.
  destruct Hc as [-> | (self & _ & ->)]; [ apply Codes_same | ].
  pose proof (looped_code_binds_statics self c) as Hbinds.
  destruct c as [params my_closure ret exn metadata body].
  apply Codes_related; [ symmetry; exact Hbinds | exact I ].
Qed.

(* The same group of bindings added to two such environments: both fail,
   or the results are two such environments. *)
Lemma eval_static_bindings_loop (vars : env) (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    statics_loop statics statics' ->
    match
      eval_static_bindings vars bindings statics,
      eval_static_bindings vars bindings statics'
    with
    | Some extended, Some extended' => statics_loop extended extended'
    | None, None => True
    | _, _ => False
    end.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics' Hstatics; simpl; [ exact Hstatics | | | ].
  - pose proof (lookup_code_loop id statics statics' Hstatics) as Hid.
    destruct (lookup id (static_code statics)),
      (lookup id (static_code statics')); try contradiction; [ exact I | ].
    apply IH. destruct Hstatics as [Hcode Hvalues]. split; [ | exact Hvalues ].
    apply Forall2_cons; [ | exact Hcode ].
    split; [ reflexivity | left; reflexivity ].
  - apply IH, Hstatics.
  - destruct Hstatics as [Hcode Hvalues]. rewrite <- Hvalues.
    destruct (lookup sym (static_values statics)); [ exact I | ].
    destruct (eval_static_const vars const) as [allocated | ]; simpl;
      [ | exact I ].
    apply IH. split; [ exact Hcode | ]. simpl. rewrite Hvalues. reflexivity.
Qed.

(* The bindings with some code made loops, added to two such
   environments. *)
Lemma eval_static_bindings_loopify (G : loopified) (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    statics_loop statics statics' ->
    match
      eval_static_bindings vars bindings statics,
      eval_static_bindings vars (map (loopify_binding G) bindings) statics'
    with
    | Some extended, Some extended' => statics_loop extended extended'
    | None, None => True
    | _, _ => False
    end.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics' Hstatics; cbn [map]; [ exact Hstatics | | | ].
  - pose proof (lookup_code_loop id statics statics' Hstatics) as Hid.
    assert (Hc : ∃ c' : code,
      loopify_binding G (Bind_code id c) = Bind_code id c' ∧ code_loops c c').
    { simpl. destruct (lookup id G) as [self | ];
        [ | exists c; split; [ reflexivity | left; reflexivity ] ].
      exists (loopify_code self c). split; [ reflexivity | ].
      unfold loopify_code. destruct (loopable self c) eqn:Hloopable;
        [ right; exists self; auto | left; reflexivity ]. }
    destruct Hc as (c' & -> & Hc). simpl.
    destruct (lookup id (static_code statics)),
      (lookup id (static_code statics')); try contradiction; [ exact I | ].
    apply IH. destruct Hstatics as [Hcode Hvalues]. split; [ | exact Hvalues ].
    apply Forall2_cons; [ | exact Hcode ].
    split; [ reflexivity | exact Hc ].
  - simpl. apply IH, Hstatics.
  - simpl. destruct Hstatics as [Hcode Hvalues]. rewrite <- Hvalues.
    destruct (lookup sym (static_values statics)); [ exact I | ].
    destruct (eval_static_const vars const) as [allocated | ]; simpl;
      [ | exact I ].
    apply IH. split; [ exact Hcode | ]. simpl. rewrite Hvalues. reflexivity.
Qed.

(* ======================================================================== *)
(* A call of code made a loop                                               *)
(* ======================================================================== *)

(* One call of code that was made a loop: the name of the loop, the code
   as it was, the closure called, and what the second run had when the
   call began: its variables, and what its return and exception
   continuations stood for. *)
Record frame : Type := Frame {
  frame_self : continuation;
  frame_code : code;
  frame_closure : value;
  frame_entry : env;
  frame_return : cont_value;
  frame_raise : cont_value
}.

Definition frame_params (F : frame) : list variable :=
  code_params (frame_code F).

Definition frame_my_closure (F : frame) : variable :=
  code_my_closure (frame_code F).

Definition frame_params_arity (F : frame) : arity :=
  code_params_arity (frame_code F).

Definition frame_ret (F : frame) : continuation :=
  code_return_continuation (frame_code F).

Definition frame_exn (F : frame) : continuation :=
  code_exn_continuation (frame_code F).

(* The body of the frame's code, or a part of it, with the self tail calls
   made jumps. *)
Definition looped (F : frame) (e : expr) : expr :=
  loop_calls (frame_my_closure F) (frame_params_arity F) (frame_ret F)
    (frame_exn F) (frame_self F) e.

Definition looped_handler (F : frame) (h : continuation_handler)
    : continuation_handler :=
  loop_calls_in_handler (frame_my_closure F) (frame_params_arity F)
    (frame_ret F) (frame_exn F) (frame_self F) h.

Definition looped_handlers (F : frame)
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k, h) => (k, looped_handler F h)) handlers.

Definition safe (F : frame) (e : expr) : bool :=
  loop_safe (frame_my_closure F) (frame_ret F) (frame_exn F) (frame_self F) e.

Definition safe_handler (F : frame) (h : continuation_handler) : bool :=
  loop_safe_handler (frame_my_closure F) (frame_ret F) (frame_exn F)
    (frame_self F) h.

(* What [loop_safe] asks of each handler of a recursive [Let_cont]. *)
Definition safe_rec (F : frame) (kh : continuation * continuation_handler)
    : bool :=
  let '(k, h) := kh in
  negb (Pos.eqb k (frame_ret F)) && negb (Pos.eqb k (frame_exn F))
  && negb (Pos.eqb k (frame_self F))
  && safe_handler F h.

(* The continuations of the second run at the entry, and the closure of
   the loop made there. *)
Definition frame_base (F : frame) : cont_env :=
  [(frame_ret F, frame_return F); (frame_exn F, frame_raise F)].

Definition frame_self_closure (F : frame) : cont_value :=
  Closure (loop_handler (frame_self F) (frame_code F)) (frame_entry F)
    (frame_base F)
    [(frame_self F, loop_handler (frame_self F) (frame_code F))].

(* The code was fit to be made a loop and binds nothing static, and the
   variables at the entry are the parameters and [my_closure], bound to
   the closure called. *)
Definition frame_wf (F : frame) : Prop :=
  loopable (frame_self F) (frame_code F) = true
  ∧ binds_statics (code_body (frame_code F)) = false
  ∧ ∀ x : variable,
      x ∉ frame_params F ->
      lookup x (frame_entry F)
      = lookup x [(frame_my_closure F, frame_closure F)].

(* In the first run, the closure called is a closure of the code. *)
Definition frame_statics (F : frame) (statics : static_env) : Prop :=
  ∃ (id : code_id) (slots : list (value_slot * value)),
    closure_of statics (frame_closure F) = Some (id, slots)
    ∧ lookup id (static_code statics) = Some (frame_code F).

Lemma frame_statics_extend (F : frame) (statics statics' : static_env) :
  statics_extend statics statics' ->
  frame_statics F statics -> frame_statics F statics'.
Proof.
  intros [Hcode Hvalues] (id & slots & Hclosure & Hid).
  exists id, slots. split; [ | apply Hcode, Hid ].
  apply (closure_of_statics_extend statics statics' (frame_closure F));
    [ | exact Hclosure ].
  intros s allocated _. apply Hvalues.
Qed.

(* What [frame_wf] gives. *)
Lemma frame_wf_facts (F : frame) :
  frame_wf F ->
  frame_my_closure F ∉ frame_params F
  ∧ frame_self F <> frame_ret F
  ∧ frame_self F <> frame_exn F
  ∧ safe F (code_body (frame_code F)) = true
  ∧ lookup (frame_my_closure F) (frame_entry F) = Some (frame_closure F).
Proof.
  intros (Hloopable & _ & Hentry). unfold loopable in Hloopable.
  apply andb_true_iff in Hloopable as [Hloopable _].
  apply andb_true_iff in Hloopable as [Hloopable _].
  apply andb_true_iff in Hloopable as [Hloopable Hsafe].
  apply andb_true_iff in Hloopable as [Hloopable Hexn].
  apply andb_true_iff in Hloopable as [Hloopable Hret].
  apply andb_true_iff in Hloopable as [Hparams _].
  apply negb_true_iff in Hparams, Hret, Hexn.
  apply Pos.eqb_neq in Hret, Hexn.
  assert (Hnotin : frame_my_closure F ∉ frame_params F).
  { intros Hin. apply among_In in Hin.
    unfold frame_my_closure, frame_params in Hin. congruence. }
  repeat split; try assumption.
  rewrite (Hentry _ Hnotin). simpl. rewrite Pos.eqb_refl. reflexivity.
Qed.

(* Code that was fit to be made a loop has distinct parameters. *)
Lemma loopable_distinct (self : continuation) (c : code) :
  loopable self c = true -> distinct (code_params c) = true.
Proof.
  unfold loopable. intros Hloopable.
  apply andb_true_iff in Hloopable as [Hloopable _].
  apply andb_true_iff in Hloopable as [Hloopable _].
  apply andb_true_iff in Hloopable as [Hloopable _].
  apply andb_true_iff in Hloopable as [Hloopable _].
  apply andb_true_iff in Hloopable as [Hloopable _].
  apply andb_true_iff in Hloopable as [_ Hdistinct]. exact Hdistinct.
Qed.

(* And is not tupled. *)
Lemma loopable_not_tupled (self : continuation) (c : code) :
  loopable self c = true -> code_is_tupled c = false.
Proof.
  unfold loopable. intros Hloopable.
  apply andb_true_iff in Hloopable as [_ Htupled].
  apply negb_true_iff in Htupled. exact Htupled.
Qed.

(* The variables of the two runs inside the loop: every name has the same
   value, and [my_closure] is the closure called. *)
Definition vars_rel (F : frame) (vars vars' : env) : Prop :=
  (∀ x : variable, lookup x vars = lookup x vars')
  ∧ lookup (frame_my_closure F) vars = Some (frame_closure F).

(* ======================================================================== *)
(* Continuations and states                                                 *)
(* ======================================================================== *)

(* What a continuation name stands for in the two runs: the same thing;
   closures of one handler under the same variables, whose continuations
   are in turn related; or closures made inside a loop, the second with
   the self tail calls of its handler made jumps, under continuations
   that are related but for [self], which in the second is the closure
   of the loop, and of which the return and exception continuations
   stand in the second for what they stood for at the entry; or related
   things, with the same extra arguments; or what is left of the same
   over-application, to return and to raise to related things. *)
Inductive cont_rel (statics : static_env) : cont_value -> cont_value -> Prop :=
  | Rel_same (c : cont_value) : cont_rel statics c c
  | Rel_out (h : continuation_handler) (vars : env) (conts conts' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    (∀ k : continuation,
      cont_rel statics (resolve k conts) (resolve k conts')) ->
    cont_rel statics (Closure h vars conts recursive)
      (Closure h vars conts' recursive)
  | Rel_in (F : frame) (h : continuation_handler) (vars vars' : env)
      (conts conts' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    frame_wf F ->
    frame_statics F statics ->
    safe_handler F h = true ->
    forallb (safe_rec F) recursive = true ->
    vars_rel F vars vars' ->
    (∀ k : continuation,
      k <> frame_self F ->
      cont_rel statics (resolve k conts) (resolve k conts')) ->
    resolve (frame_self F) conts' = frame_self_closure F ->
    resolve (frame_ret F) conts' = resolve (frame_ret F) (frame_base F) ->
    resolve (frame_exn F) conts' = resolve (frame_exn F) (frame_base F) ->
    cont_rel statics (Closure h vars conts recursive)
      (Closure (looped_handler F h) vars' conts' (looped_handlers F recursive))
  | Rel_extra_args (c c' : cont_value) (extra : list value) :
    cont_rel statics c c' ->
    cont_rel statics (Extra_args c extra) (Extra_args c' extra)
  | Rel_over_application (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    cont_rel statics return_to return_to' ->
    cont_rel statics exn_to exn_to' ->
    cont_rel statics
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Lemma with_extra_args_rel (statics : static_env) (c c' : cont_value)
    (extra : list value) :
  cont_rel statics c c' ->
  cont_rel statics (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hrel. destruct extra; simpl;
    [ exact Hrel | apply Rel_extra_args, Hrel ].
Qed.

(* The continuations of the two runs inside the loop. *)
Definition in_conts (F : frame) (statics : static_env)
    (conts conts' : cont_env) : Prop :=
  (∀ k : continuation,
    k <> frame_self F ->
    cont_rel statics (resolve k conts) (resolve k conts'))
  ∧ resolve (frame_self F) conts' = frame_self_closure F
  ∧ resolve (frame_ret F) conts' = resolve (frame_ret F) (frame_base F)
  ∧ resolve (frame_exn F) conts' = resolve (frame_exn F) (frame_base F).

(* What was known of the static environment still holds when it has
   grown. *)
Lemma cont_rel_extend (statics statics' : static_env) (c c' : cont_value) :
  statics_extend statics statics' ->
  cont_rel statics c c' -> cont_rel statics' c c'.
Proof.
  intros Hextend Hrel.
  induction Hrel
    as [ c | h vars conts conts' recursive _ IH
       | F h vars vars' conts conts' recursive Hwf Hstatics Hh Hrec Hvars _ IH
           Hself Hret Hexn
       | c c' extra _ IH
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' _ IHreturn _ IHraise ].
  - apply Rel_same.
  - apply Rel_out, IH.
  - apply Rel_in; try assumption.
    exact (frame_statics_extend F statics statics' Hextend Hstatics).
  - apply Rel_extra_args, IH.
  - apply Rel_over_application; assumption.
Qed.

Lemma in_conts_extend (F : frame) (statics statics' : static_env)
    (conts conts' : cont_env) :
  statics_extend statics statics' ->
  in_conts F statics conts conts' -> in_conts F statics' conts conts'.
Proof.
  intros Hextend (Hconts & Hrest). split; [ | exact Hrest ].
  intros k Hk. exact (cont_rel_extend _ _ _ _ Hextend (Hconts k Hk)).
Qed.

Inductive state_loops : state -> state -> Prop :=
  | Loops_out (e : expr) (vars : env) (conts conts' : cont_env)
      (statics statics' : static_env) (hp : heap) (rd : read_record) :
    statics_loop statics statics' ->
    (∀ k : continuation,
      cont_rel statics (resolve k conts) (resolve k conts')) ->
    state_loops (State e vars conts statics hp rd)
      (State e vars conts' statics' hp rd)
  | Loops_in (F : frame) (e : expr) (vars vars' : env)
      (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
        (rd : read_record) :
    statics_loop statics statics' ->
    frame_wf F ->
    frame_statics F statics ->
    safe F e = true ->
    vars_rel F vars vars' ->
    in_conts F statics conts conts' ->
    state_loops (State e vars conts statics hp rd)
      (State (looped F e) vars' conts' statics' hp rd).

(* Two continuations bound, one after the other, to related things. *)
Lemma pair_conts_rel (statics : static_env) (a b : continuation)
    (r r' x x' : cont_value) :
  cont_rel statics r r' ->
  (a <> b -> cont_rel statics x x') ->
  ∀ k : continuation,
    cont_rel statics (resolve k [(a, r); (b, x)])
      (resolve k [(a, r'); (b, x')]).
Proof.
  intros Hr Hx k.
  destruct (Pos.eq_dec k a) as [-> | Hne];
    [ rewrite !resolve_head; exact Hr | ].
  rewrite !(resolve_other k a) by exact Hne.
  destruct (Pos.eq_dec k b) as [-> | Hne'];
    [ rewrite !resolve_head; apply Hx; congruence | ].
  rewrite !(resolve_other k b) by exact Hne'. apply Rel_same.
Qed.

(* A name that is none of those of the handlers is not bound by them. *)
Lemma lookup_safe_rec (F : frame) (k : continuation)
    (recursive : list (continuation * continuation_handler)) :
  forallb (safe_rec F) recursive = true ->
  k = frame_ret F ∨ k = frame_exn F ∨ k = frame_self F ->
  lookup k recursive = None.
Proof.
  intros Hrec Hk.
  induction recursive as [ | [k' h] recursive IH]; simpl; [ reflexivity | ].
  simpl in Hrec. apply andb_true_iff in Hrec as [Hhead Hrest].
  apply andb_true_iff in Hhead as [Hhead _].
  apply andb_true_iff in Hhead as [Hhead Hself].
  apply andb_true_iff in Hhead as [Hret Hexn].
  apply negb_true_iff in Hret, Hexn, Hself.
  apply Pos.eqb_neq in Hret, Hexn, Hself.
  destruct (Pos.eqb_spec k k') as [-> | _]; [ | exact (IH Hrest) ].
  destruct Hk as [Hk | [Hk | Hk]]; congruence.
Qed.

Lemma safe_rec_lookup (F : frame) (k : continuation)
    (h : continuation_handler)
    (recursive : list (continuation * continuation_handler)) :
  forallb (safe_rec F) recursive = true ->
  lookup k recursive = Some h ->
  safe_handler F h = true.
Proof.
  intros Hrec Hlookup. rewrite forallb_forall in Hrec.
  specialize (Hrec (k, h) (lookup_in _ _ _ Hlookup)). simpl in Hrec.
  apply andb_true_iff in Hrec as [_ Hh]. exact Hh.
Qed.

(* The closures of the handlers of a recursive [Let_cont] inside a
   loop. *)
Lemma in_conts_rec (F : frame) (statics : static_env)
    (recursive : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  frame_wf F ->
  frame_statics F statics ->
  forallb (safe_rec F) recursive = true ->
  vars_rel F vars vars' ->
  in_conts F statics conts conts' ->
  in_conts F statics (rec_closures recursive vars conts ++ conts)
    (rec_closures (looped_handlers F recursive) vars' conts' ++ conts').
Proof.
  intros Hwf Hstatics Hrec Hvars (Hconts & Hself & Hret & Hexn).
  assert (Hnone :
    ∀ k : continuation,
      k = frame_ret F ∨ k = frame_exn F ∨ k = frame_self F ->
      resolve k (rec_closures (looped_handlers F recursive) vars' conts'
                 ++ conts')
      = resolve k conts').
  { intros k Hk. rewrite resolve_rec_closures. unfold looped_handlers.
    unfold looped_handler. rewrite lookup_loop_handlers,
      (lookup_safe_rec F k recursive Hrec Hk).
    reflexivity. }
  split; [ | rewrite !Hnone by tauto; auto ].
  intros k Hk. rewrite !resolve_rec_closures. unfold looped_handlers.
  unfold looped_handler at 1. rewrite lookup_loop_handlers.
  destruct (lookup k recursive) as [h | ] eqn:Hlookup; simpl;
    [ | exact (Hconts k Hk) ].
  apply (Rel_in statics F h vars vars' conts conts' recursive); try assumption.
  exact (safe_rec_lookup F k h recursive Hrec Hlookup).
Qed.

(* ======================================================================== *)
(* Applying continuations and calling code                                  *)
(* ======================================================================== *)

(* The values at an exit are the same in the two runs. *)
Definition same_values (last : state) (vs vs' : list value) : Prop := vs' = vs.

Lemma answered_one (s1 t t1 : state) :
  step t = Step t1 ->
  state_loops s1 t1 ->
  answered state_loops same_values (Step s1) t.
Proof.
  intros Hstep Hrel. exists 0, t1.
  split; [ simpl; rewrite Hstep; reflexivity | exact Hrel ].
Qed.

(* What two steps come to: the first is undefined; or both exit alike;
   or both step, the second then going on for some steps, to related
   states. *)
Definition result_loops (result result' : step_result) : Prop :=
  match result with
  | Undefined => True
  | Exit k vs hp rd => result' = Exit k vs hp rd
  | Step s1 =>
    ∃ (t1 : state) (n : nat) (t2 : state),
      result' = Step t1 ∧ run n t1 = Step t2 ∧ state_loops s1 t2
  end.

(* An application that does not name [self] does not apply it. *)
Lemma not_among_apply_cont (self : continuation) (ac : apply_cont) :
  negb (among self (apply_cont_continuations ac)) = true ->
  ac_continuation ac <> self.
Proof.
  intros Hsafe Heq. apply negb_true_iff in Hsafe.
  unfold apply_cont_continuations, among in Hsafe. simpl in Hsafe.
  rewrite Heq, Pos.eqb_refl in Hsafe. discriminate Hsafe.
Qed.

(* The looped code, entered with distinct parameters bound to values of
   their kinds: two steps bind [self] and jump to it, to the looped body
   under variables that agree with those of the entry. *)
Lemma enter_loop (self : continuation) (c : code) (vs : list value)
    (base entry : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  binds_statics (code_body c) = false ->
  distinct (code_params c) = true ->
  map kind_of_value vs = code_params_kinds c ->
  bind_params (code_params c) vs base = Some entry ->
  ∃ entry' : env,
    run 2 (State (code_body (looped_code self c)) entry conts statics hp rd)
    = Step
        (State
           (loop_calls (code_my_closure c) (code_params_arity c)
              (code_return_continuation c) (code_exn_continuation c) self
              (code_body c))
           entry'
           ((self,
             Closure (loop_handler self c) entry conts
               [(self, loop_handler self c)])
            :: conts)
           statics hp rd)
    ∧ ∀ x : variable, lookup x entry = lookup x entry'.
Proof.
  intros Hbinds Hdistinct Hkinds Hbind.
  destruct (rebind_params (code_params c) vs base entry Hbind)
    as (vs' & entry' & Hvs' & Hbind' & Hext).
  rewrite (eval_params_bound _ vs base entry Hdistinct Hbind) in Hvs'.
  injection Hvs' as <-.
  pose proof (eval_params_bound _ vs base entry Hdistinct Hbind) as Hvs.
  apply kinds_eqb_eq in Hkinds.
  exists entry'. split; [ | exact Hext ].
  destruct c as [params my_closure ret exn metadata body]. simpl in *.
  rewrite step_let_cont_recursive. unfold handlers_bind_statics. simpl.
  rewrite loop_calls_binds_statics, Hbinds. simpl.
  rewrite step_apply_cont. simpl. rewrite Hvs.
  unfold apply_continuation. rewrite resolve_head. unfold loop_handler.
  simpl. rewrite Hkinds, Hbind'. reflexivity.
Qed.

(* The same full call, of code and of what the second static environment
   has for it, returning to related continuations: the second run goes
   on for no step, or for two if the code was made a loop. *)
Lemma call_code_loops (statics statics' : static_env) (hp : heap)
    (rd : read_record)
    (closure : value)
    (called : code_id) (slots : list (value_slot * value)) (c c' : code)
    (vs : list value) (args_arity return_arity : arity)
    (r r' x x' : cont_value) :
  statics_loop statics statics' ->
  closure_of statics closure = Some (called, slots) ->
  lookup called (static_code statics) = Some c ->
  code_loops c c' ->
  cont_rel statics r r' ->
  cont_rel statics x x' ->
  result_loops
    (call_code closure c vs args_arity return_arity r x statics hp rd)
    (call_code closure c' vs args_arity return_arity r' x' statics' hp rd).
Proof.
  intros Hstatics Hclosure Hc Hcode Hr Hx.
  destruct Hcode as [-> | (self & Hloopable & ->)]; unfold call_code.
  - destruct (binds_statics (code_body c)); [ exact I | ].
    match goal with
    | |- context [if negb ?agrees then _ else _] =>
      destruct (negb agrees); [ exact I | ]
    end.
    destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
      as [entry | ]; [ | exact I ].
    cbn [result_loops]. eexists _, 0, _.
    split; [ reflexivity | ]. split; [ reflexivity | ].
    apply Loops_out; [ exact Hstatics | ].
    apply pair_conts_rel; [ exact Hr | intros _; exact Hx ].
  - destruct (looped_code_fields self c)
      as (Hsame_params & Hsame_closure & Hsame_ret & Hsame_exn
          & Hsame_params_arity & Hsame_result_arity).
    rewrite looped_code_binds_statics, Hsame_params_arity,
      Hsame_result_arity, Hsame_params, Hsame_closure, Hsame_ret, Hsame_exn.
    destruct (binds_statics (code_body c)) eqn:Hbinds; [ exact I | ].
    match goal with
    | |- context [if negb ?agrees then _ else _] =>
      destruct (negb agrees) eqn:Hagrees; [ exact I | ]
    end.
    destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
      as [entry | ] eqn:Hbind; [ | exact I ].
    destruct (enter_loop self c vs _ entry
                [(code_return_continuation c, r');
                 (code_exn_continuation c, x')]
                statics' hp rd Hbinds (loopable_distinct self c Hloopable)
                (agreed_kinds vs args_arity return_arity c Hagrees) Hbind)
      as (entry' & Hrun & Hext).
    cbn [result_loops]. eexists _, 2, _.
    split; [ reflexivity | ]. split; [ exact Hrun | ].
    set (F := Frame self c closure entry r' x').
    assert (Hwf : frame_wf F).
    { split; [ exact Hloopable | ]. split; [ exact Hbinds | ].
      intros y Hy.
      exact (lookup_bind_params_other (code_params c) vs _ entry y Hbind Hy). }
    destruct (frame_wf_facts F Hwf)
      as (Hparams & Hret & Hexn & Hsafe & Hmy_closure).
    apply (Loops_in F); try assumption.
    + exists called, slots. split; assumption.
    + split; [ exact Hext | exact Hmy_closure ].
    + split; [ | split; [ | split ] ].
      * intros k Hk. rewrite (resolve_other k self) by exact Hk.
        apply pair_conts_rel; [ exact Hr | intros _; exact Hx ].
      * apply resolve_head.
      * apply resolve_other. intros Heq. apply Hret. symmetry. exact Heq.
      * apply resolve_other. intros Heq. apply Hexn. symmetry. exact Heq.
Qed.

(* A call is of the same shape under the two static environments. *)
Lemma shape_of_call_loops (statics statics' : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity) :
  statics_loop statics statics' ->
  shapes_related code_loops
    (shape_of_call statics callee kind vs args_arity return_arity)
    (shape_of_call statics' callee kind vs args_arity return_arity).
Proof.
  intros Hstatics. apply shape_of_call_related.
  - symmetry.
    exact
      (closure_of_related (ignoring_binders any_body) statics statics'
         (called_closure callee) (statics_loop_related _ _ Hstatics)).
  - intros called slots _. unfold codes_agree.
    pose proof (lookup_code_loop called statics statics' Hstatics) as Hcode.
    destruct (lookup called (static_code statics)) as [c | ],
      (lookup called (static_code statics')) as [c' | ];
      try contradiction; [ | exact I ].
    split; [ exact Hcode | ].
    destruct Hcode as [-> | (self & _ & ->)]; [ repeat split | ].
    destruct (looped_code_fields self c)
      as (_ & _ & _ & _ & Hparams & Hresult).
    split; [ exact Hparams | ]. split; [ exact Hresult | ].
    destruct c; reflexivity.
  - intros tuple rest p _. destruct Hstatics as [_ Hvalues].
    symmetry. exact (eval_prim_same_static_values _ _ p [tuple] Hvalues).
Qed.

(* Calls of the same shape under the two static environments, returning
   to related continuations that take a partial application to related
   results. *)
Lemma run_call_loops (statics statics' : static_env) (hp : heap)
    (rd : read_record)
    (kind : call_kind)
    (shape shape' : option call_shape) (return_arity : arity)
    (r r' x x' : cont_value) :
  statics_loop statics statics' ->
  shapes_related code_loops shape shape' ->
  shape_calls statics kind shape ->
  cont_rel statics r r' ->
  cont_rel statics x x' ->
  (∀ partial : value,
    result_loops (apply_cont_value r [partial] statics hp rd)
      (apply_cont_value r' [partial] statics' hp rd)) ->
  result_loops
    (run_call (fun partial => apply_cont_value r [partial] statics hp rd) shape
       return_arity r x statics hp rd)
    (run_call (fun partial => apply_cont_value r' [partial] statics' hp rd)
       shape'
       return_arity r' x' statics' hp rd).
Proof.
  intros Hstatics Hshapes Hcalls Hr Hx Hpartial.
  destruct Hshapes
    as [ | closure c c' vs args_arity Hcode
       | closure c c' vs remaining args_arity remaining_arity Hcode
       | partial ];
    cbn [run_call].
  - exact I.
  - destruct Hcalls as (called & slots & Hclosure & Hc & _).
    exact
      (call_code_loops statics statics' _ rd_ closure called slots c c' vs
         args_arity return_arity r r' x x' Hstatics Hclosure Hc Hcode Hr Hx).
  - destruct Hcalls as (called & slots & Hclosure & Hc & _).
    assert (Hresult : code_result_arity c' = code_result_arity c).
    { destruct Hcode as [-> | (self & _ & ->)]; [ reflexivity | ].
      apply looped_code_fields. }
    rewrite Hresult.
    apply (call_code_loops statics statics' _ rd_ closure called slots);
      try assumption.
    apply Rel_over_application; assumption.
  - apply Hpartial.
Qed.

(* Related continuations applied to the same values. *)
Lemma apply_cont_loops (statics statics' : static_env) (hp : heap)
    (rd : read_record)
    (c c' : cont_value) :
  statics_loop statics statics' ->
  cont_rel statics c c' ->
  ∀ vs : list value,
    result_loops (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' vs statics' hp rd).
Proof.
  intros Hstatics Hrel.
  induction Hrel
    as [ c | [params metadata handler] vars conts conts' recursive Hconts _
       | F [params metadata handler] vars vars' conts conts' recursive Hwf
           Hfstatics Hh Hrec Hvars Hconts _ Hself Hret Hexn
       | c c' extra _ IH
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise _ ];
    intros vs.
  - revert vs.
    induction c
      as [[params metadata handler] vars conts recursive | k | c IH extra
         | remaining remaining_arity return_arity return_to IHreturn exn_to
             _];
      intros vs.
    + simpl.
      destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
        [ | exact I ].
      destruct (bind_params params vs vars) as [bound | ]; [ | exact I ].
      eexists _, 0, _. split; [ reflexivity | ]. split; [ reflexivity | ].
      apply Loops_out; [ exact Hstatics | intros k; apply Rel_same ].
    + simpl. reflexivity.
    + simpl. apply IH.
    + rewrite !apply_cont_value_over_application.
      destruct vs as [ | v [ | v' vs]]; try exact I.
      unfold call_function.
      apply (run_call_loops statics statics' _ rd_ Indirect);
        [ exact Hstatics | apply shape_of_call_loops, Hstatics
        | apply shape_of_call_calls | apply Rel_same | apply Rel_same | ].
      intros partial. apply IHreturn.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs vars) as [bound | ]; [ | exact I ].
    eexists _, 0, _. split; [ reflexivity | ]. split; [ reflexivity | ].
    apply Loops_out; [ exact Hstatics | ].
    intros k. rewrite !resolve_rec_closures.
    destruct (lookup k recursive) as [h | ];
      [ apply Rel_out, Hconts | apply Hconts ].
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs vars) as [bound | ] eqn:Hbind;
      [ | exact I ].
    destruct (bind_params_ext params vs vars vars' bound (proj1 Hvars) Hbind)
      as (bound' & Hbind' & Hext).
    rewrite Hbind'. eexists _, 0, _.
    split; [ reflexivity | ]. split; [ reflexivity | ].
    unfold safe_handler in Hh. simpl in Hh.
    apply andb_true_iff in Hh as [Hparams Hhandler].
    apply (Loops_in F); try assumption.
    + split; [ exact Hext | ].
      rewrite (lookup_bind_params_other params vs vars bound _ Hbind);
        [ exact (proj2 Hvars) | ].
      intros Hin. apply among_In in Hin. rewrite Hin in Hparams.
      discriminate.
    + apply in_conts_rec; try assumption. repeat split; assumption.
  - simpl. apply IH.
  - rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; try exact I.
    unfold call_function.
    apply (run_call_loops statics statics' _ rd_ Indirect);
      [ exact Hstatics | apply shape_of_call_loops, Hstatics
      | apply shape_of_call_calls | exact Hreturn | exact Hraise | ].
    intros partial. apply IHreturn.
Qed.

(* The same call under the two static environments, returning to
   related continuations. *)
Lemma call_loops (statics statics' : static_env) (hp : heap) (rd : read_record)
    (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (r r' x x' : cont_value) :
  statics_loop statics statics' ->
  cont_rel statics r r' ->
  cont_rel statics x x' ->
  result_loops
    (call_function callee kind vs args_arity return_arity r x statics hp rd)
    (call_function callee kind vs args_arity return_arity r' x' statics' hp rd).
Proof.
  intros Hstatics Hr Hx. unfold call_function.
  apply (run_call_loops statics statics' _ rd_ kind);
    [ exact Hstatics | apply shape_of_call_loops, Hstatics
    | apply shape_of_call_calls | exact Hr | exact Hx | ].
  intros partial.
  exact (apply_cont_loops statics statics' _ rd_ r r' Hstatics Hr [partial]).
Qed.

(* ======================================================================== *)
(* The simulation                                                           *)
(* ======================================================================== *)

Lemma vars_rel_let (F : frame) (x : variable) (v : value) (vars vars' : env) :
  x <> frame_my_closure F ->
  vars_rel F vars vars' ->
  vars_rel F ((x, v) :: vars) ((x, v) :: vars').
Proof.
  intros Hx [Hagree Hclosure]. split.
  - intros y. simpl. destruct (Pos.eqb y x); [ reflexivity | apply Hagree ].
  - simpl. destruct (Pos.eqb_spec (frame_my_closure F) x) as [Heq | _];
      [ symmetry in Heq; contradiction | exact Hclosure ].
Qed.

(* The closure of the handler of a [Let_cont] inside a loop. *)
Lemma in_conts_let_cont (F : frame) (statics : static_env) (k : continuation)
    (h : continuation_handler) (vars vars' : env) (conts conts' : cont_env) :
  frame_wf F ->
  frame_statics F statics ->
  k <> frame_ret F ->
  k <> frame_exn F ->
  k <> frame_self F ->
  safe_handler F h = true ->
  vars_rel F vars vars' ->
  in_conts F statics conts conts' ->
  in_conts F statics ((k, Closure h vars conts []) :: conts)
    ((k, Closure (looped_handler F h) vars' conts' []) :: conts').
Proof.
  intros Hwf Hstatics Hret Hexn Hself Hh Hvars (Hconts & Hs & Hr & Hx).
  split; [ | split; [ | split ] ].
  - intros k' Hk'. destruct (Pos.eq_dec k' k) as [-> | Hne].
    + rewrite !resolve_head.
      apply (Rel_in statics F h vars vars' conts conts' []); auto.
    + rewrite !(resolve_other k' k) by exact Hne. apply Hconts, Hk'.
  - rewrite resolve_other by congruence. exact Hs.
  - rewrite resolve_other by congruence. exact Hr.
  - rewrite resolve_other by congruence. exact Hx.
Qed.

(* A self tail call inside the loop: the first run calls the code again,
   the second jumps to the loop, and the two are inside the loop again. *)
Lemma self_tail_call_loops (F : frame) (kind : call_kind) (vs : list value)
    (args_arity return_arity : arity) (conts conts' : cont_env)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  statics_loop statics statics' ->
  frame_wf F ->
  frame_statics F statics ->
  in_conts F statics conts conts' ->
  length args_arity = length (frame_params_arity F) ->
  match
    call_function (frame_closure F) kind vs args_arity return_arity
      (resolve (frame_ret F) conts) (resolve (frame_exn F) conts) statics hp rd
  with
  | Undefined => True
  | Exit _ _ _ _ => False
  | Step s1 =>
    ∃ t1 : state,
      apply_cont_value (resolve (frame_self F) conts') vs statics' hp rd
        = Step t1
      ∧ state_loops s1 t1
  end.
Proof.
  intros Hstatics Hwf Hfstatics Hin Hfull.
  unfold frame_params_arity in Hfull.
  pose proof Hfstatics as (id & slots & Hclosure & Hid).
  pose proof Hwf as (Hloopable & Hbinds & Hentry).
  destruct (frame_wf_facts F Hwf) as (Hparams & Hret & Hexn & Hsafe & Hmy).
  pose proof Hin as (Hconts & Hself & Hr & Hx).
  unfold call_function.
  rewrite
    (shape_of_call_closure statics (frame_closure F) kind vs args_arity
       return_arity (closure_not_partial statics _ _ Hclosure)).
  unfold shape_of_closure_call. rewrite Hclosure.
  destruct
    (match kind with
     | Direct named => Pos.eqb id named
     | Indirect => true
     end);
    [ | exact I ].
  rewrite Hid. unfold arguments_for_code.
  rewrite (loopable_not_tupled _ _ Hloopable). cbv beta iota.
  unfold shape_of_code_call. cbv zeta.
  rewrite Hfull, Nat.eqb_refl. cbn [run_call].
  unfold call_code. rewrite Hbinds.
  match goal with
  | |- context [if negb ?agrees then _ else _] =>
    destruct (negb agrees) eqn:Hagrees; [ exact I | ]
  end.
  apply agreed_kinds, kinds_eqb_eq in Hagrees.
  destruct
    (bind_params (code_params (frame_code F)) vs
       [(code_my_closure (frame_code F), frame_closure F)])
    as [entry | ] eqn:Hbind; [ | exact I ].
  destruct (bind_params_defined _ vs _ (frame_entry F) entry Hbind)
    as [bound' Hbind'].
  rewrite Hself. unfold frame_self_closure, loop_handler.
  cbn [apply_cont_value hm_params_kinds make_handler_metadata].
  rewrite Hagrees, Hbind'. eexists. split; [ reflexivity | ].
  apply (Loops_in F (code_body (frame_code F)) entry bound');
    try assumption.
  - split.
    + apply (bind_params_agree_outside _ vs _ _ entry bound' Hbind Hbind').
      intros y Hy. symmetry. apply Hentry, Hy.
    + rewrite (lookup_bind_params_other _ vs _ entry _ Hbind Hparams).
      simpl. rewrite Pos.eqb_refl. reflexivity.
  - split; [ | split; [ | split ] ].
    + intros k Hk. rewrite resolve_rec_closures. simpl.
      destruct (Pos.eqb_spec k (frame_self F)) as [Heq | _];
        [ contradiction | ].
      apply pair_conts_rel.
      * pose proof (Hconts (frame_ret F) (not_eq_sym Hret)) as Hreturn.
        rewrite Hr in Hreturn. unfold frame_base in Hreturn.
        rewrite resolve_head in Hreturn. exact Hreturn.
      * intros Hne.
        pose proof (Hconts (frame_exn F) (not_eq_sym Hexn)) as Hraise.
        rewrite Hx in Hraise. unfold frame_base in Hraise.
        rewrite resolve_other, resolve_head in Hraise
          by (intros Heq; apply Hne; symmetry; exact Heq).
        exact Hraise.
    + rewrite resolve_rec_closures. simpl. rewrite Pos.eqb_refl.
      reflexivity.
    + rewrite resolve_rec_closures. simpl.
      destruct (Pos.eqb_spec (frame_ret F) (frame_self F)) as [Heq | _];
        [ symmetry in Heq; contradiction | reflexivity ].
    + rewrite resolve_rec_closures. simpl.
      destruct (Pos.eqb_spec (frame_exn F) (frame_self F)) as [Heq | _];
        [ symmetry in Heq; contradiction | reflexivity ].
Qed.

Lemma loops_step (s t : state) :
  state_loops s t -> answered state_loops same_values (step s) t.
Proof.
  intros
    [ e vars conts conts' statics statics' hp rd Hstatics Hconts
    | F e vars vars' conts conts' statics statics' hp rd Hstatics Hwf
        Hfstatics Hsafe Hvars Hin ].
  - (* outside a loop *)
    pose proof (statics_loop_related statics statics' Hstatics) as Hrelated.
    destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ].
    + (* Let *)
      rewrite step_let_exec.
      destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hd;
        [ | exact I ].
      cbv beta iota.
      destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind; [ | exact I ].
      apply
        (answered_one _ _
           (State body ((x, v) :: vars) conts' statics' hp' rd')).
      * rewrite step_let_exec,
          <- (exec_named_related (ignoring_binders any_body) statics statics'
                hp rd_ vars d Hrelated),
          Hd.
        cbv beta iota. rewrite Hkind. reflexivity.
      * apply Loops_out; assumption.
    + (* Let_static *)
      rewrite step_let_static.
      pose proof
        (eval_static_bindings_loop vars bindings statics statics' Hstatics)
        as Hbindings.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hextended; [ | exact I ].
      destruct (eval_static_bindings vars bindings statics')
        as [extended' | ] eqn:Hextended'; [ | contradiction ].
      apply (answered_one _ _ (State body vars conts' extended' hp rd)).
      * rewrite step_let_static, Hextended'. reflexivity.
      * apply Loops_out; [ exact Hbindings | ].
        intros k. apply (cont_rel_extend statics extended); [ | apply Hconts ].
        exact
          (eval_static_bindings_extend vars bindings statics extended
             Hextended).
    + (* Let_cont_non_recursive *)
      rewrite step_let_cont_non_recursive.
      apply
        (answered_one _ _
           (State body vars ((k, Closure h vars conts' []) :: conts')
              statics' hp rd)).
      * apply step_let_cont_non_recursive.
      * apply Loops_out; [ exact Hstatics | ].
        intros k'. destruct (Pos.eq_dec k' k) as [-> | Hne].
        -- rewrite !resolve_head. apply Rel_out, Hconts.
        -- rewrite !(resolve_other k' k) by exact Hne. apply Hconts.
    + (* Let_cont_recursive *)
      rewrite step_let_cont_recursive.
      destruct (handlers_bind_statics handlers) eqn:Hbinds; [ exact I | ].
      apply
        (answered_one _ _
           (State body vars (rec_closures handlers vars conts' ++ conts')
              statics' hp rd)).
      * rewrite step_let_cont_recursive, Hbinds. reflexivity.
      * apply Loops_out; [ exact Hstatics | ].
        intros k. rewrite !resolve_rec_closures.
        destruct (lookup k handlers) as [h | ];
          [ apply Rel_out, Hconts | apply Hconts ].
    + (* Apply *)
      rewrite step_apply.
      destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
        [ | exact I ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
        [ | exact I ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
        eqn:Hextra; [ | exact I ].
      pose proof
        (call_loops statics statics' hp rd callee (ap_call_kind ap) vs
           (ap_args_arity ap) (ap_return_arity ap) _ _ _ _ Hstatics
           (Hconts (ap_continuation ap))
           (with_extra_args_rel statics _ _ extra
              (Hconts (ap_exn_continuation ap))))
        as Hcall.
      destruct
        (call_function callee (ap_call_kind ap) vs (ap_args_arity ap)
           (ap_return_arity ap) (resolve (ap_continuation ap) conts)
           (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
           statics hp rd)
        as [s1 | k1 vs1 | ]; cbn [result_loops] in Hcall.
      * destruct Hcall as (t1 & n & t2 & Hcall & Hrun & Hrel).
        exists n, t2. split; [ | exact Hrel ].
        simpl. rewrite step_apply, Hcallee, Hargs, Hextra, Hcall. exact Hrun.
      * exists vs1. split; [ | reflexivity ].
        rewrite step_apply, Hcallee, Hargs, Hextra. exact Hcall.
      * exact I.
    + (* Apply_cont *)
      rewrite step_apply_cont.
      destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
        [ | exact I ].
      unfold apply_continuation.
      pose proof
        (apply_cont_loops statics statics' hp rd _ _ Hstatics
           (Hconts (ac_continuation ac)) vs)
        as Happly.
      destruct (apply_cont_value (resolve (ac_continuation ac) conts) vs
                  statics hp rd)
        as [s1 | k1 vs1 | ].
      * destruct Happly as (t1 & n & t2 & Happly & Hrun & Hrel).
        exists n, t2. split; [ | exact Hrel ].
        simpl. rewrite step_apply_cont, Hargs. unfold apply_continuation.
        rewrite Happly. exact Hrun.
      * exists vs1. split; [ | reflexivity ].
        rewrite step_apply_cont, Hargs. exact Happly.
      * exact I.
    + (* Switch *)
      rewrite step_switch.
      destruct (eval_simple vars scrutinee) as [v | ] eqn:Hv; [ | exact I ].
      destruct (naked_immediate_of_value v) as [i | ] eqn:Hi; [ | exact I ].
      destruct (find_arm i arms) as [ac | ] eqn:Hfind; [ | exact I ].
      apply
        (answered_one _ _ (State (Apply_cont ac) vars conts' statics' hp rd)).
      * rewrite step_switch, Hv, Hi, Hfind. reflexivity.
      * apply Loops_out; assumption.
    + rewrite step_invalid. exact I.
  - (* inside a loop *)
    pose proof (statics_loop_related statics statics' Hstatics) as Hrelated.
    pose proof Hvars as [Hagree Hmy].
    destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      unfold safe in Hsafe; simpl in Hsafe; unfold looped; cbn [loop_calls].
    + (* Let *)
      apply andb_true_iff in Hsafe as [Hx Hbody].
      apply negb_true_iff in Hx. apply Pos.eqb_neq in Hx.
      rewrite step_let_exec.
      destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hd;
        [ | exact I ].
      cbv beta iota.
      destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind; [ | exact I ].
      apply
        (answered_one _ _
           (State (looped F body) ((x, v) :: vars') conts' statics' hp' rd')).
      * rewrite step_let_exec,
          (exec_named_agree statics' hp rd vars vars' d (fun y _ => Hagree y)),
          <- (exec_named_related (ignoring_binders any_body) statics statics'
                hp rd_ vars d Hrelated),
          Hd.
        cbv beta iota. rewrite Hkind. reflexivity.
      * apply (Loops_in F); try assumption. apply vars_rel_let; assumption.
    + (* Let_static *)
      rewrite step_let_static.
      pose proof
        (eval_static_bindings_loop vars bindings statics statics' Hstatics)
        as Hbindings.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hextended; [ | exact I ].
      destruct (eval_static_bindings vars bindings statics')
        as [extended' | ] eqn:Hextended'; [ | contradiction ].
      pose proof
        (eval_static_bindings_extend vars bindings statics extended Hextended)
        as Hextend.
      apply
        (answered_one _ _ (State (looped F body) vars' conts' extended' hp rd)).
      * rewrite step_let_static,
          (eval_static_bindings_agree vars vars' bindings
             (fun y _ => Hagree y)),
          Hextended'.
        reflexivity.
      * apply (Loops_in F); try assumption.
        -- exact (frame_statics_extend F statics extended Hextend Hfstatics).
        -- exact (in_conts_extend F statics extended conts conts' Hextend Hin).
    + (* Let_cont_non_recursive *)
      apply andb_true_iff in Hsafe as [Hsafe Hbody].
      apply andb_true_iff in Hsafe as [Hsafe Hh].
      apply andb_true_iff in Hsafe as [Hsafe Hself].
      apply andb_true_iff in Hsafe as [Hret Hexn].
      apply negb_true_iff in Hret, Hexn, Hself.
      apply Pos.eqb_neq in Hret, Hexn, Hself.
      rewrite step_let_cont_non_recursive.
      apply
        (answered_one _ _
           (State (looped F body) vars'
              ((k, Closure (looped_handler F h) vars' conts' []) :: conts')
              statics' hp rd)).
      * apply step_let_cont_non_recursive.
      * apply (Loops_in F); try assumption.
        apply in_conts_let_cont; assumption.
    + (* Let_cont_recursive *)
      apply andb_true_iff in Hsafe as [Hrec Hbody].
      rewrite step_let_cont_recursive.
      destruct (handlers_bind_statics handlers) eqn:Hbinds; [ exact I | ].
      apply
        (answered_one _ _
           (State (looped F body) vars'
              (rec_closures (looped_handlers F handlers) vars' conts'
               ++ conts')
              statics' hp rd)).
      * rewrite step_let_cont_recursive, loop_handlers_bind_statics, Hbinds.
        reflexivity.
      * apply (Loops_in F); try assumption.
        apply in_conts_rec; assumption.
    + (* Apply *)
      apply andb_true_iff in Hsafe as [Hk Hx].
      apply negb_true_iff in Hk, Hx. apply Pos.eqb_neq in Hk, Hx.
      rewrite step_apply.
      destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
        [ | exact I ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
        [ | exact I ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
        eqn:Hextra; [ | exact I ].
      destruct
        (is_self_tail_call (frame_my_closure F) (frame_params_arity F)
           (frame_ret F) (frame_exn F) ap)
        eqn:Htail.
      * (* a self tail call *)
        unfold is_self_tail_call in Htail.
        apply andb_true_iff in Htail as [Htail Hno_extra].
        apply andb_true_iff in Htail as [Htail Hexn].
        apply andb_true_iff in Htail as [Htail Hret].
        apply andb_true_iff in Htail as [Hvar Hfull].
        apply Pos.eqb_eq in Hret, Hexn. apply Nat.eqb_eq in Hfull.
        destruct (ap_callee ap) as [y | sym | c]; try discriminate.
        apply Pos.eqb_eq in Hvar. subst y.
        simpl in Hcallee. rewrite Hmy in Hcallee. injection Hcallee as <-.
        unfold no_exn_extra_args in Hno_extra.
        unfold exn_extra_simples in Hextra.
        destruct (ap_exn_extra_args ap); [ | discriminate Hno_extra ].
        simpl in Hextra. injection Hextra as <-. cbn [with_extra_args].
        rewrite Hret, Hexn.
        pose proof
          (self_tail_call_loops F (ap_call_kind ap) vs (ap_args_arity ap)
             (ap_return_arity ap) conts conts' statics statics' hp rd Hstatics
               Hwf
             Hfstatics Hin Hfull)
          as Hcall.
        destruct
          (call_function (frame_closure F) (ap_call_kind ap) vs
             (ap_args_arity ap) (ap_return_arity ap)
             (resolve (frame_ret F) conts) (resolve (frame_exn F) conts)
             statics hp rd)
          as [s1 | k1 vs1 | ]; [ | contradiction | exact I ].
        destruct Hcall as (t1 & Happly & Hrel).
        apply (answered_one _ _ t1); [ | exact Hrel ].
        rewrite step_apply_cont. cbn [ac_args ac_continuation].
        rewrite (eval_simples_agree vars vars' (ap_args ap)
                   (fun y _ => Hagree y)),
          Hargs.
        exact Happly.
      * (* another call *)
        destruct Hin as (Hconts & _).
        pose proof
          (call_loops statics statics' hp rd callee (ap_call_kind ap) vs
             (ap_args_arity ap) (ap_return_arity ap) _ _ _ _ Hstatics
             (Hconts _ Hk)
             (with_extra_args_rel statics _ _ extra (Hconts _ Hx)))
          as Hcall.
        destruct
          (call_function callee (ap_call_kind ap) vs (ap_args_arity ap)
             (ap_return_arity ap) (resolve (ap_continuation ap) conts)
             (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
             statics hp rd)
          as [s1 | k1 vs1 | ]; cbn [result_loops] in Hcall.
        -- destruct Hcall as (t1 & n & t2 & Hcall & Hrun & Hrel).
           exists n, t2. split; [ | exact Hrel ].
           simpl.
           rewrite step_apply,
             (eval_simple_agree vars vars' _ (fun y _ => Hagree y)),
             (eval_simples_agree vars vars' (ap_args ap)
                (fun y _ => Hagree y)),
             (eval_simples_agree vars vars' (exn_extra_simples ap)
                (fun y _ => Hagree y)),
             Hcallee, Hargs, Hextra, Hcall.
           exact Hrun.
        -- exists vs1. split; [ | reflexivity ].
           rewrite step_apply,
             (eval_simple_agree vars vars' _ (fun y _ => Hagree y)),
             (eval_simples_agree vars vars' (ap_args ap)
                (fun y _ => Hagree y)),
             (eval_simples_agree vars vars' (exn_extra_simples ap)
                (fun y _ => Hagree y)),
             Hcallee, Hargs, Hextra.
           exact Hcall.
        -- exact I.
    + (* Apply_cont *)
      apply not_among_apply_cont in Hsafe.
      rewrite step_apply_cont.
      destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
        [ | exact I ].
      unfold apply_continuation.
      destruct Hin as (Hconts & _).
      pose proof
        (apply_cont_loops statics statics' hp rd _ _ Hstatics (Hconts _ Hsafe)
           vs)
        as Happly.
      destruct (apply_cont_value (resolve (ac_continuation ac) conts) vs
                  statics hp rd)
        as [s1 | k1 vs1 | ].
      * destruct Happly as (t1 & n & t2 & Happly & Hrun & Hrel).
        exists n, t2. split; [ | exact Hrel ].
        simpl.
        rewrite step_apply_cont,
          (eval_simples_agree vars vars' _ (fun y _ => Hagree y)), Hargs.
        unfold apply_continuation. rewrite Happly. exact Hrun.
      * exists vs1. split; [ | reflexivity ].
        rewrite step_apply_cont,
          (eval_simples_agree vars vars' _ (fun y _ => Hagree y)), Hargs.
        exact Happly.
      * exact I.
    + (* Switch *)
      rewrite step_switch.
      destruct (eval_simple vars scrutinee) as [v | ] eqn:Hv; [ | exact I ].
      destruct (naked_immediate_of_value v) as [i | ] eqn:Hi; [ | exact I ].
      destruct (find_arm i arms) as [ac | ] eqn:Hfind; [ | exact I ].
      apply
        (answered_one _ _
           (State (looped F (Apply_cont ac)) vars' conts' statics' hp rd)).
      * rewrite step_switch,
          (eval_simple_agree vars vars' _ (fun y _ => Hagree y)), Hv, Hi,
          Hfind.
        reflexivity.
      * apply (Loops_in F); try assumption.
        rewrite forallb_forall in Hsafe.
        exact (Hsafe (i, ac) (find_arm_in i arms ac Hfind)).
    + rewrite step_invalid. exact I.
Qed.

Lemma loops_refines (s t : state) : state_loops s t -> refines s t.
Proof.
  apply (stuttering_refines state_loops same_values loops_step).
  intros last vs vs' Hsame. exact Hsame.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The [Let_static] that binds the code made loops. *)
Lemma let_static_loopify (G : loopified) (bindings : list static_binding)
    (body : expr) :
  expr_refines (Let_static bindings body)
    (Let_static (map (loopify_binding G) bindings) body).
Proof.
  intros vars conts statics hp rd. apply let_static_refines_step.
  pose proof
    (eval_static_bindings_loopify G vars bindings statics statics
       (statics_loop_refl statics))
    as Hbindings.
  destruct (eval_static_bindings vars bindings statics) as [extended | ];
    [ | exact I ].
  destruct (eval_static_bindings vars (map (loopify_binding G) bindings)
              statics)
    as [extended' | ]; [ | contradiction ].
  apply loops_refines.
  apply Loops_out; [ exact Hbindings | intros k; apply Rel_same ].
Qed.

Fixpoint loopify_unit_refines_aux (G : loopified) (e : expr) :
  expr_refines e (loopify_unit G e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try apply expr_refines_refl.
  - exact (plug_refines (Let_body x kind d Hole) _ _ (not_many _)
      (loopify_unit_refines_aux G body)).
  - apply (expr_refines_trans _ (Let_static bindings (loopify_unit G body)));
      [ | apply let_static_loopify ].
    exact (plug_refines (Let_static_body bindings Hole) _ _ (not_many _)
      (loopify_unit_refines_aux G body)).
  - apply (expr_refines_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (loopify_unit G handler)) body)).
    + exact (plug_refines
        (Let_cont_non_recursive_handler k params metadata Hole body) _ _
        (not_many _) (loopify_unit_refines_aux G handler)).
    + exact (plug_refines
        (Let_cont_non_recursive_body k
           (Handler params metadata (loopify_unit G handler)) Hole)
        _ _ (not_many _) (loopify_unit_refines_aux G body)).
  - exact (plug_refines (Let_cont_recursive_body handlers Hole) _ _
      (not_many _) (loopify_unit_refines_aux G body)).
Qed.

Theorem loopify_unit_refines : loopify_unit_refines_statement.
Proof. exact loopify_unit_refines_aux. Qed.
