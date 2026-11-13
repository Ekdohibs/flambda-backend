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


(* Proofs for FlambdaLiftAroundSpec.v. Nothing here needs review.

   The two runs are in step, but that where a group is put the second
   run takes a step more, to bind it. The second expression is the
   first with its [Let]s made aliases and its groups put; values are
   related by FlambdaLiftSpec.value_lifted, and the static environments
   as in FlambdaLiftFromCodeProofs.v with no code rewritten. Beside
   that the states have what the check has gone by: constants that are
   known ([known_valid]: each symbol is bound, in the second run, to
   what its constant is in the variables of the second run as they are
   now), for which the check passes of the expression; and the same of
   every handler that a continuation keeps, in the variables that it
   keeps. A step that binds a variable keeps what the check kept
   ([forget]). So where a [Let] that is made an alias is executed, the
   symbol is bound to the constant of the [Let] evaluated in the
   variables of that moment, which is what the [Let] makes, lifted.

   The second run is that of a well-formed unit: it never binds a
   symbol twice (FlambdaWellFormedProofs.well_formed_never_faults) and
   reads no variable that has no value
   (FlambdaWellFormedProofs.well_formed_names_bound), so a group that is
   put is evaluated. That the values at the exit are equal is from the
   invariant of the first run that FlambdaLiftProofs.v has for the
   check of the exits ([plain_state]).

   Without the test of the exits (FlambdaLiftObservedSpec.v), the two
   runs are not started from what may follow a unit: such a term may
   read a variable that has no value, and the step lemma here asks
   that the second run reads none. The two units are run on their own
   to their exits, which are from two states related
   ([stuttering_reaches]), and what may follow is then the same term in
   the two runs, on values lifted and under static environments with
   the same code: the simulation of FlambdaLiftProofs.v with nothing
   lifted ([observer_applied_lifted]), in which no group is put. That
   gives the statement about the two exits ([unit_around_worlds]), and
   FlambdaObserversSpec.under_observers_worlds_statement gives from it
   the one about the runs from what may follow a unit
   ([unit_around_observed]); the code of the unit is closed because
   that of the unit rewritten is ([lift_around_code_closed]). *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveProofs
  FlambdaBindersSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaSimulationProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaScopeSpec FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaClosuresSpec FlambdaClosuresProofs FlambdaStaticProofs
  FlambdaUnusedStaticsSpec FlambdaImportsSpec FlambdaExitWorldSpec
  FlambdaExitWorldProofs FlambdaRunUnderSpec FlambdaObserversSpec
  FlambdaObserversProofs FlambdaUnitFromProofs FlambdaLiftSpec
  FlambdaLiftProofs FlambdaLiftFromCodeSpec FlambdaLiftFromCodeProofs
  FlambdaLiftAroundSpec.
From OxCamlFlambda2 Require Import FlambdaAlphaProofs.
Import ListNotations.

(* ======================================================================== *)
(* With nothing to do the rewrites change nothing                           *)
(* ======================================================================== *)

Fixpoint lift_around_nothing (at_path : path) (e : expr) {struct e} :
  lift_around [] [] at_path e = e
with lift_around_handler_nothing (at_path : path)
    (h : continuation_handler) {struct h} :
  lift_around_handler [] [] at_path h = h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite lift_around_nothing. reflexivity.
    + rewrite lift_around_nothing. reflexivity.
    + rewrite lift_around_handler_nothing, lift_around_nothing. reflexivity.
    + rewrite lift_around_nothing. f_equal.
      generalize (xI at_path).
      induction handlers as [ | [k1 h1] handlers IH]; intros q;
        [ reflexivity | ].
      cbn [rec_handlers_map].
      rewrite lift_around_handler_nothing, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite lift_around_nothing. reflexivity.
Qed.

Lemma lift_around_handlers_nothing :
  ∀ (handlers : list (continuation * continuation_handler)) (q : path),
    rec_handlers_map (lift_around_handler [] []) q handlers = handlers.
Proof.
  induction handlers as [ | [k h] handlers IH]; intros q; [ reflexivity | ].
  cbn [rec_handlers_map]. rewrite lift_around_handler_nothing, IH.
  reflexivity.
Qed.

Fixpoint around_ok_nothing (known : known_constants) (at_path : path)
    (e : expr) {struct e} :
  around_ok [] [] known at_path e = true
with handler_around_ok_nothing (known : known_constants) (at_path : path)
    (h : continuation_handler) {struct h} :
  handler_around_ok [] [] known at_path h = true.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply around_ok_nothing.
    + apply around_ok_nothing.
    + rewrite handler_around_ok_nothing, around_ok_nothing. reflexivity.
    + rewrite lift_around_handlers_nothing, Bool.eqb_reflx, around_ok_nothing.
      rewrite andb_true_r. simpl.
      generalize (xI at_path).
      induction handlers as [ | [k1 h1] handlers IH]; intros q;
        [ reflexivity | ].
      cbn [rec_handlers_forall].
      rewrite handler_around_ok_nothing, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl. apply around_ok_nothing.
Qed.

(* So with no [Let] to make an alias and no place for a group, the unit
   is left as it is, whether its conditions hold or not. *)
Lemma lift_around_unit_nothing (u : unit_interface) (e : expr) :
  match lift_around_unit u [] [] e with
  | Some e' => e'
  | None => e
  end
  = e.
Proof.
  unfold lift_around_unit, groups_of. cbn [map]. cbv zeta.
  rewrite (lift_around_nothing root e).
  destruct
    (exits_plain [] e && around_ok [] [] [] root e && well_formed_unit u e);
    reflexivity.
Qed.

(* The rewrite of FlambdaLiftFromCodeSpec.v with no code named and no
   [Let] given. *)
Fixpoint lifted_out_bindings_nothing (at_path : path) (e : expr) {struct e} :
  lifted_out_bindings [] at_path e = []
with handler_lifted_out_bindings_nothing (at_path : path)
    (h : continuation_handler) {struct h} :
  handler_lifted_out_bindings [] at_path h = [].
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply lifted_out_bindings_nothing.
    + apply lifted_out_bindings_nothing.
    + rewrite lifted_out_bindings_nothing,
        handler_lifted_out_bindings_nothing.
      reflexivity.
    + rewrite lifted_out_bindings_nothing. simpl.
      generalize (xI at_path).
      induction handlers as [ | [k1 h1] handlers IH]; intros q;
        [ reflexivity | ].
      cbn [rec_handlers_bindings].
      rewrite handler_lifted_out_bindings_nothing, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply lifted_out_bindings_nothing.
Qed.

Fixpoint lift_from_code_nothing (at_path : path) (e : expr) {struct e} :
  lift_from_code [] [] at_path e = e
with handler_lift_from_code_nothing (at_path : path)
    (h : continuation_handler) {struct h} :
  handler_lift_from_code [] [] at_path h = h
with binding_lift_from_code_nothing (binding : static_binding)
    {struct binding} :
  binding_lift_from_code [] binding = [binding].
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite lift_from_code_nothing. reflexivity.
    + rewrite lift_from_code_nothing. f_equal.
      induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
      rewrite (binding_lift_from_code_nothing binding), IH. reflexivity.
    + rewrite handler_lift_from_code_nothing, lift_from_code_nothing.
      reflexivity.
    + rewrite lift_from_code_nothing. f_equal.
      generalize (xI at_path).
      induction handlers as [ | [k1 h1] handlers IH]; intros q;
        [ reflexivity | ].
      cbn [rec_handlers_map].
      rewrite handler_lift_from_code_nothing, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite lift_from_code_nothing. reflexivity.
  - destruct binding as [id [params my_closure ret exn metadata body] | id
      | s const];
      simpl; try reflexivity.
    rewrite lifted_out_bindings_nothing, lift_from_code_nothing. reflexivity.
Qed.

Lemma bindings_lift_from_code_nothing_at_all (bindings : list static_binding) :
  flat_map (binding_lift_from_code []) bindings = bindings.
Proof.
  induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
  rewrite binding_lift_from_code_nothing, IH. reflexivity.
Qed.

Lemma code_lifted_out_nothing (id : code_id) (c : code) :
  code_lifted_out [] id c = c.
Proof.
  destruct c as [params my_closure ret exn metadata body]. simpl.
  rewrite lift_from_code_nothing. reflexivity.
Qed.

(* ======================================================================== *)
(* Evaluation under variables that agree                                    *)
(* ======================================================================== *)

Lemma eval_simple_same (vars vars1 : env) (s : simple) :
  (∀ x : variable,
    x ∈ free_variables_of_simple s -> lookup x vars1 = lookup x vars) ->
  eval_simple vars1 s = eval_simple vars s.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hagree; try reflexivity.
  apply Hagree. left. reflexivity.
Qed.

Lemma eval_simples_same (vars vars1 : env) (args : list simple) :
  (∀ x : variable,
    x ∈ free_variables_of_simples args -> lookup x vars1 = lookup x vars) ->
  eval_simples vars1 args = eval_simples vars args.
Proof.
  unfold free_variables_of_simples.
  induction args as [ | s args IH]; intros Hagree; simpl; [ reflexivity | ].
  rewrite (eval_simple_same vars vars1 s), IH; [ reflexivity | | ];
    intros x Hx; apply Hagree; simpl; apply in_or_app;
    [ right | left ]; exact Hx.
Qed.

Lemma eval_value_slots_same (vars vars1 : env)
    (slots : list (value_slot * simple)) :
  (∀ x : variable,
    x ∈ free_variables_of_value_slots slots ->
    lookup x vars1 = lookup x vars) ->
  eval_value_slots vars1 slots = eval_value_slots vars slots.
Proof.
  unfold free_variables_of_value_slots.
  induction slots as [ | [slot s] slots IH]; intros Hagree; simpl;
    [ reflexivity | ].
  rewrite (eval_simple_same vars vars1 s), IH; [ reflexivity | | ];
    intros x Hx; apply Hagree; simpl; apply in_or_app;
    [ right | left ]; exact Hx.
Qed.

Lemma eval_static_const_same (vars vars1 : env) (const : static_const) :
  (∀ x : variable,
    x ∈ free_variables_of_static_const const ->
    lookup x vars1 = lookup x vars) ->
  eval_static_const vars1 const = eval_static_const vars const.
Proof.
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl; intros Hagree.
  - rewrite (eval_value_slots_same vars vars1 slots Hagree). reflexivity.
  - rewrite (eval_value_slots_same vars vars1 slots Hagree). reflexivity.
  - rewrite (eval_simples_same vars vars1 fields Hagree). reflexivity.
  - rewrite (eval_simple_same vars vars1 contents Hagree). reflexivity.
  - reflexivity.
  - rewrite (eval_simples_same vars vars1 elements Hagree). reflexivity.
Qed.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* Each symbol known is bound, in the second run, to what its constant
   is in the variables given. *)
Definition known_valid (statics' : static_env) (vars' : env)
    (known : known_constants) : Prop :=
  ∀ (s : symbol) (const : static_const),
    (s, const) ∈ known ->
    ∃ allocated : statically_allocated_value,
      eval_static_const vars' const = Some allocated
      ∧ lookup s (static_values statics') = Some allocated.

Lemma known_valid_nil (statics' : static_env) (vars' : env) :
  known_valid statics' vars' [].
Proof. intros s const []. Qed.

Lemma known_valid_extend (statics' statics'' : static_env) (vars' : env)
    (known : known_constants) :
  statics_extend statics' statics'' ->
  known_valid statics' vars' known ->
  known_valid statics'' vars' known.
Proof.
  intros Hextend Hvalid s const Hin.
  destruct (Hvalid s const Hin) as (allocated & Hconst & Hs).
  exists allocated. split; [ exact Hconst | exact (proj2 Hextend s _ Hs) ].
Qed.

Lemma known_valid_app (statics' : static_env) (vars' : env)
    (known known1 : known_constants) :
  known_valid statics' vars' known ->
  known_valid statics' vars' known1 ->
  known_valid statics' vars' (known ++ known1).
Proof.
  intros H H1 s const Hin. apply in_app_or in Hin as [Hin | Hin];
    [ exact (H s const Hin) | exact (H1 s const Hin) ].
Qed.

(* What is known of variables that agree outside those bound again. *)
Lemma known_valid_forget (statics' : static_env) (vars' vars1 : env)
    (bound : list variable) (known : known_constants) :
  (∀ x : variable, x ∉ bound -> lookup x vars1 = lookup x vars') ->
  known_valid statics' vars' known ->
  known_valid statics' vars1 (forget bound known).
Proof.
  intros Hagree Hvalid s const Hin. unfold forget in Hin.
  apply filter_In in Hin as [Hin Hkept]. cbn [snd] in Hkept.
  destruct (Hvalid s const Hin) as (allocated & Hconst & Hs).
  exists allocated. split; [ | exact Hs ].
  rewrite <- Hconst. apply eval_static_const_same. intros x Hx.
  apply Hagree. intros Hbound.
  apply negb_true_iff in Hkept.
  assert (Hexists :
    existsb (fun x => among x bound) (free_variables_of_static_const const)
    = true).
  { apply existsb_exists. exists x.
    split; [ exact Hx | exact (proj2 (among_In _ _) Hbound) ]. }
  exact (eq_true_false_abs _ Hexists Hkept).
Qed.

Lemma known_valid_let (statics' : static_env) (vars' : env) (x : variable)
    (v' : value) (known : known_constants) :
  known_valid statics' vars' known ->
  known_valid statics' ((x, v') :: vars') (forget [x] known).
Proof.
  apply known_valid_forget. intros y Hy. apply lookup_other.
  intros ->. apply Hy. left. reflexivity.
Qed.

(* A variable that is no parameter is bound, once the parameters are, to
   what it was. *)
Lemma lookup_past_params (params : list variable) :
  ∀ (vs : list value) (vars bound : env) (x : variable),
    bind_params params vs vars = Some bound ->
    x ∉ params ->
    lookup x bound = lookup x vars.
Proof.
  induction params as [ | param params IH]; intros [ | v vs] vars bound x;
    simpl; try discriminate.
  - intros [= <-] _. reflexivity.
  - intros Hbind Hnotin.
    rewrite (IH vs _ bound x Hbind (fun Hin => Hnotin (or_intror Hin))).
    apply lookup_other. intros ->. apply Hnotin. left. reflexivity.
Qed.

Lemma known_valid_params (statics' : static_env) (vars' bound' : env)
    (params : list variable) (vs' : list value) (known : known_constants) :
  bind_params params vs' vars' = Some bound' ->
  known_valid statics' vars' known ->
  known_valid statics' bound' (forget params known).
Proof.
  intros Hbind. apply known_valid_forget. intros y Hy.
  exact (lookup_past_params params vs' vars' bound' y Hbind Hy).
Qed.

Lemma constant_known_in (known : known_constants) (s : symbol)
    (const : static_const) :
  constant_known known (Bind_symbol s const) = true -> (s, const) ∈ known.
Proof.
  simpl. intros Hknown.
  apply existsb_exists in Hknown as ([s1 const1] & Hin & Hsame).
  cbn [fst snd] in Hsame. apply andb_true_iff in Hsame as [Hs Hconst].
  apply Pos.eqb_eq in Hs. subst s1.
  destruct (static_const_eq_dec const const1) as [-> | ];
    [ exact Hin | discriminate ].
Qed.

(* ======================================================================== *)
(* A [Let] made an alias of a symbol that is known                          *)
(* ======================================================================== *)

(* The value that the first run makes is the symbol's, lifted: the
   symbol is bound to the constant of the [Let] as it is in the
   variables of the second run, which are those of the first, lifted. *)
Lemma known_alias (target : lift_to) (d : named)
    (bindings : list static_binding) (s : symbol)
    (statics statics' : static_env) (vars vars' : env) (v : value)
    (known : known_constants) :
  lift_named target d = Some (bindings, s) ->
  forallb (constant_known known) bindings = true ->
  known_valid statics' vars' known ->
  env_lifted statics' vars vars' ->
  eval_named statics vars d = Some v ->
  value_lifted statics' v (Value_symbol s).
Proof.
  intros Hlift Hknown Hvalid Hvars Heval.
  destruct target as [sym | syms].
  - destruct d as [ | p args | code slots | | | | ]; simpl in Hlift;
      try discriminate.
    2: { (* A closure of a set of one function. *)
      injection Hlift as <- <-. simpl in Heval.
      destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
        [ | discriminate ].
      injection Heval as <-.
      destruct (eval_value_slots_lifted statics' vars vars' slots Hvars vs
                  Hslots)
        as (vs' & Hslots' & Hvs).
      cbn [forallb] in Hknown. apply andb_true_iff in Hknown as [Hknown _].
      destruct (Hvalid _ _ (constant_known_in _ _ _ Hknown))
        as (allocated & Hconst & Hsym).
      simpl in Hconst. rewrite Hslots' in Hconst. simpl in Hconst.
      injection Hconst as <-.
      exact (Lifted_to_unnamed_closure _ code vs vs' sym Hsym Hvs). }
    destruct p as [p | p | p | p | rp]; simpl in Hlift; try discriminate.
    + (* A boxed number. *)
      destruct p; simpl in Hlift; try discriminate.
      destruct args as [ | contents [ | arg args]]; simpl in Hlift;
        try discriminate.
      injection Hlift as <- <-. simpl in Heval.
      destruct (eval_simple vars contents) as [w | ] eqn:Hcontents;
        [ | discriminate ].
      rewrite eval_prim_unary in Heval. simpl in Heval.
      destruct w as [number | | | | | | | | ]; try discriminate.
      destruct (is_number_of_kind kind number); [ | discriminate ].
      injection Heval as <-.
      destruct (eval_simple_lifted statics' vars vars' contents _ Hvars
                  Hcontents)
        as (w' & Hcontents' & Hw).
      inversion Hw; subst w'.
      cbn [forallb] in Hknown. apply andb_true_iff in Hknown as [Hknown _].
      destruct (Hvalid _ _ (constant_known_in _ _ _ Hknown))
        as (allocated & Hconst & Hsym).
      simpl in Hconst. rewrite Hcontents' in Hconst. simpl in Hconst.
      injection Hconst as <-.
      apply Lifted_to_boxed_number. exact Hsym.
    + (* A block, or an immutable array. *)
      destruct p as [[tag] | akind]; simpl in Hlift.
      * injection Hlift as <- <-.
        simpl in Heval.
        destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
          [ | discriminate ].
        rewrite eval_prim_variadic in Heval. simpl in Heval.
        destruct
          (forallb (fun v => kind_eqb (kind_of_value v) Value) vs
           && forallb (bound_if_symbol statics) vs);
          [ | discriminate ].
        injection Heval as <-.
        destruct (eval_simples_lifted statics' vars vars' args Hvars vs Hargs)
          as (vs' & Hargs' & Hvs).
        cbn [forallb] in Hknown. apply andb_true_iff in Hknown as [Hknown _].
        destruct (Hvalid _ _ (constant_known_in _ _ _ Hknown))
          as (allocated & Hconst & Hsym).
        simpl in Hconst. rewrite Hargs' in Hconst. simpl in Hconst.
        injection Hconst as <-.
        exact (Lifted_to_block _ tag vs vs' sym Hsym Hvs).
      * injection Hlift as <- <-.
        simpl in Heval.
        destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
          [ | discriminate ].
        rewrite eval_prim_variadic in Heval. simpl in Heval.
        unfold array_made in Heval.
        destruct (forallb (array_element_ok akind) vs); [ | discriminate ].
        simpl in Heval.
        destruct (forallb (bound_if_symbol statics) vs); [ | discriminate ].
        injection Heval as <-.
        destruct (eval_simples_lifted statics' vars vars' args Hvars vs Hargs)
          as (vs' & Hargs' & Hvs).
        cbn [forallb] in Hknown. apply andb_true_iff in Hknown as [Hknown _].
        destruct (Hvalid _ _ (constant_known_in _ _ _ Hknown))
          as (allocated & Hconst & Hsym).
        simpl in Hconst. rewrite Hargs' in Hconst. simpl in Hconst.
        injection Hconst as <-.
        exact (Lifted_to_array _ vs vs' sym Hsym Hvs).
  - (* A closure of a set. *)
    destruct d as [ | | | fslot decls slots | | | ]; simpl in Hlift;
      try discriminate.
    destruct (lookup fslot syms) as [s1 | ] eqn:Hsym; [ | discriminate ].
    destruct (list_eq_dec Pos.eq_dec (map fst syms) (map fst decls))
      as [Hsyms | ];
      [ | discriminate ].
    injection Hlift as <- <-. simpl in Heval.
    destruct (lookup fslot decls) as [code | ]; [ | discriminate ].
    simpl in Heval.
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    injection Heval as <-.
    destruct (eval_value_slots_lifted statics' vars vars' slots Hvars vs
                Hslots)
      as (vs' & Hslots' & Hvs).
    apply (Lifted_to_closure statics' fslot decls vs vs' s1 syms Hsym Hsyms);
      [ | exact Hvs ].
    intros slot s0 Hin. rewrite forallb_forall in Hknown.
    assert (Hbinding :
      Bind_symbol s0 (Static_const_closure_of_set slot syms decls slots)
      ∈ bind_closures syms decls slots).
    { unfold bind_closures. apply in_map_iff. exists (slot, s0).
      split; [ reflexivity | exact Hin ]. }
    destruct (Hvalid _ _ (constant_known_in _ _ _ (Hknown _ Hbinding)))
      as (allocated & Hconst & Hs0).
    simpl in Hconst.
    destruct (lookup slot decls) as [code0 | ]; [ | discriminate Hconst ].
    simpl in Hconst. rewrite Hslots' in Hconst. simpl in Hconst.
    injection Hconst as <-. exact Hs0.
Qed.

(* ======================================================================== *)
(* Groups of static bindings                                                *)
(* ======================================================================== *)

(* A group that is evaluated binds each of its symbols to what its
   constant is. *)
Lemma eval_static_bindings_known (vars' : env) :
  ∀ (bindings : list static_binding) (statics' extended' : static_env),
    eval_static_bindings vars' bindings statics' = Some extended' ->
    known_valid extended' vars' (constants_of_group bindings).
Proof.
  induction bindings as [ | binding bindings IH];
    intros statics' extended' Heval.
  - apply known_valid_nil.
  - destruct binding as [id c | id | sym const]; simpl in Heval |- *.
    + destruct (lookup id (static_code statics')); [ discriminate | ].
      exact (IH _ _ Heval).
    + exact (IH _ _ Heval).
    + destruct (lookup sym (static_values statics')) eqn:Hsym;
        [ discriminate | ].
      destruct (eval_static_const vars' const) as [allocated | ] eqn:Hconst;
        [ | discriminate ].
      simpl in Heval. intros s c [Heq | Hin].
      * injection Heq as <- <-. exists allocated.
        split; [ exact Hconst | ].
        apply (proj2 (eval_static_bindings_extend vars' bindings _ extended'
                 Heval)).
        apply lookup_head.
      * exact (IH _ _ Heval s c Hin).
Qed.

Lemma eval_simple_bound (vars : env) (s : simple) :
  (∀ x : variable, x ∈ free_variables_of_simple s -> is_bound x vars = true)
  ->
  ∃ v : value, eval_simple vars s = Some v.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hbound; eauto.
  specialize (Hbound x (or_introl eq_refl)). unfold is_bound in Hbound.
  destruct (lookup x vars) as [v | ]; [ eauto | discriminate ].
Qed.

Lemma eval_simples_bound (vars : env) (args : list simple) :
  (∀ x : variable, x ∈ free_variables_of_simples args -> is_bound x vars = true)
  ->
  ∃ vs : list value, eval_simples vars args = Some vs.
Proof.
  unfold free_variables_of_simples.
  induction args as [ | s args IH]; intros Hbound; simpl; [ eauto | ].
  destruct (eval_simple_bound vars s) as [v ->].
  { intros x Hx. apply Hbound. simpl. apply in_or_app. left. exact Hx. }
  destruct IH as [vs ->]; [ | simpl; eauto ].
  intros x Hx. apply Hbound. simpl. apply in_or_app. right. exact Hx.
Qed.

Lemma eval_value_slots_bound (vars : env) (slots : list (value_slot * simple)) :
  (∀ x : variable,
    x ∈ free_variables_of_value_slots slots -> is_bound x vars = true) ->
  ∃ vs : list (value_slot * value), eval_value_slots vars slots = Some vs.
Proof.
  unfold free_variables_of_value_slots.
  induction slots as [ | [slot s] slots IH]; intros Hbound; simpl; [ eauto | ].
  destruct (eval_simple_bound vars s) as [v ->].
  { intros x Hx. apply Hbound. simpl. apply in_or_app. left. exact Hx. }
  destruct IH as [vs ->]; [ | simpl; eauto ].
  intros x Hx. apply Hbound. simpl. apply in_or_app. right. exact Hx.
Qed.

(* A constant that may be put has a value where its variables have. *)
Lemma total_const_evaluates (vars : env) (const : static_const) :
  total_const const = true ->
  (∀ x : variable,
    x ∈ free_variables_of_static_const const -> is_bound x vars = true) ->
  ∃ allocated : statically_allocated_value,
    eval_static_const vars const = Some allocated.
Proof.
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl; intros Htotal Hbound.
  - destruct (eval_value_slots_bound vars slots Hbound) as [vs ->].
    simpl. eauto.
  - destruct (lookup fslot decls); [ | discriminate ].
    destruct (eval_value_slots_bound vars slots Hbound) as [vs ->].
    simpl. eauto.
  - destruct (eval_simples_bound vars fields Hbound) as [vs ->]. simpl. eauto.
  - destruct (eval_simple_bound vars contents Hbound) as [v ->]. simpl. eauto.
  - eauto.
  - destruct (eval_simples_bound vars elements Hbound) as [vs ->].
    simpl. eauto.
Qed.

(* A group that may be put is evaluated, where its symbols are distinct
   and not bound and its variables have values; it binds no code. *)
Lemma eval_putable_group (vars' : env) :
  ∀ (bindings : list static_binding) (statics' : static_env),
    putable_group bindings = true ->
    distinct (flat_map symbols_of_binding bindings) = true ->
    (∀ s : symbol,
      s ∈ flat_map symbols_of_binding bindings ->
      lookup s (static_values statics') = None) ->
    (∀ x : variable,
      x ∈ variables_read (Let_static bindings Invalid) ->
      is_bound x vars' = true) ->
    ∃ extended' : static_env,
      eval_static_bindings vars' bindings statics' = Some extended'
      ∧ static_code extended' = static_code statics'.
Proof.
  induction bindings as [ | binding bindings IH];
    intros statics' Hputable Hdistinct Hfresh Hbound.
  - simpl. eauto.
  - destruct binding as [id c | id | sym const]; simpl in Hputable;
      try discriminate.
    apply andb_true_iff in Hputable as [Htotal Hputable].
    simpl in Hdistinct, Hfresh, Hbound |- *.
    rewrite (Hfresh sym (or_introl eq_refl)).
    destruct (total_const_evaluates vars' const Htotal) as [allocated ->].
    { intros x Hx. apply Hbound. apply in_or_app. left. exact Hx. }
    simpl. apply andb_true_iff in Hdistinct as [Hnotin Hdistinct].
    destruct
      (IH
         (Static_env (static_code statics')
            ((sym, allocated) :: static_values statics'))
         Hputable Hdistinct)
      as (extended' & Heval & Hcode).
    + intros s Hin. cbn [static_values].
      rewrite lookup_other; [ apply Hfresh; right; exact Hin | ].
      intros ->. apply negb_true_iff in Hnotin.
      apply (proj2 (among_In _ _)) in Hin.
      exact (eq_true_false_abs _ Hin Hnotin).
    + intros x Hx. apply Hbound. apply in_or_app. right. exact Hx.
    + exists extended'. split; [ exact Heval | exact Hcode ].
Qed.

(* ======================================================================== *)
(* Continuations and states                                                 *)
(* ======================================================================== *)

(* A continuation of the first run and one of the second: the same way
   out, or closures of a handler and of that handler rewritten, with
   lifted environments, the check passing of the handler for constants
   that are known in the variables kept; or what is left of the same
   over-application. The handlers of a recursive group are the same in
   both. *)
#[local] Set Warnings "-register-all".
Inductive cont_around (statics' : static_env)
    : cont_value -> cont_value -> Prop :=
  | Outer_around (k : continuation) :
    cont_around statics' (Outer k) (Outer k)
  | Closure_around (lets : list (path * lift_to))
      (groups : list (path * list static_binding)) (known : known_constants)
      (at_path : path) (h : continuation_handler) (vars vars' : env)
      (conts conts' : cont_env) (group_lets : list (path * lift_to))
      (group_groups : list (path * list static_binding)) (q : path)
      (recursive : list (continuation * continuation_handler)) :
    env_lifted statics' vars vars' ->
    Forall2
      (fun kc kc' =>
         fst kc = fst kc' ∧ cont_around statics' (snd kc) (snd kc'))
      conts conts' ->
    known_valid statics' vars' known ->
    handler_around_ok lets groups known at_path h = true ->
    rec_handlers_forall (handler_around_ok group_lets group_groups known) q
      recursive
    = true ->
    cont_around statics' (Closure h vars conts recursive)
      (Closure (lift_around_handler lets groups at_path h) vars' conts'
         (rec_handlers_map (lift_around_handler group_lets group_groups) q
            recursive))
  | Extra_args_around (c c' : cont_value) (extra extra' : list value) :
    cont_around statics' c c' ->
    Forall2 (value_lifted statics') extra extra' ->
    cont_around statics' (Extra_args c extra) (Extra_args c' extra')
  | Over_application_around (remaining remaining' : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    Forall2 (value_lifted statics') remaining remaining' ->
    cont_around statics' return_to return_to' ->
    cont_around statics' exn_to exn_to' ->
    cont_around statics'
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining' remaining_arity return_arity return_to'
         exn_to').

Definition conts_around (statics' : static_env) (conts conts' : cont_env)
    : Prop :=
  Forall2
    (fun kc kc' => fst kc = fst kc' ∧ cont_around statics' (snd kc) (snd kc'))
    conts conts'.

Fixpoint cont_around_extend (statics' statics'' : static_env)
    (Hextend : statics_extend statics' statics'') (c c' : cont_value)
    (H : cont_around statics' c c') {struct H} :
  cont_around statics'' c c'.
Proof.
  destruct H
    as [ k
       | lets groups known at_path h vars vars' conts conts' group_lets
           group_groups q recursive Hvars Hconts Hvalid Hok Hgroup
       | c0 c0' extra extra' Hc0 Hextra
       | remaining remaining' remaining_arity return_arity return_to
           return_to' exn_to exn_to' Hremaining Hreturn Hexn ].
  - apply Outer_around.
  - apply (Closure_around statics'' lets groups known);
      [ exact (env_lifted_extend _ _ _ _ Hextend Hvars) |
      | exact (known_valid_extend _ _ _ _ Hextend Hvalid) | exact Hok
      | exact Hgroup ].
    clear Hvars Hvalid. revert conts conts' Hconts. fix IH 3.
    intros conts conts' [ | kc kc' rest rest' [Hkey Hhead] Htail];
      constructor;
      [ split;
        [ exact Hkey | exact (cont_around_extend _ _ Hextend _ _ Hhead) ]
      | exact (IH _ _ Htail) ].
  - apply Extra_args_around;
      [ exact (cont_around_extend _ _ Hextend _ _ Hc0)
      | exact (values_lifted_extend _ _ _ _ Hextend Hextra) ].
  - apply Over_application_around;
      [ exact (values_lifted_extend _ _ _ _ Hextend Hremaining)
      | exact (cont_around_extend _ _ Hextend _ _ Hreturn)
      | exact (cont_around_extend _ _ Hextend _ _ Hexn) ].
Qed.

Lemma with_extra_args_around (statics' : static_env) (c c' : cont_value)
    (extra extra' : list value) :
  cont_around statics' c c' ->
  Forall2 (value_lifted statics') extra extra' ->
  cont_around statics' (with_extra_args c extra) (with_extra_args c' extra').
Proof.
  intros Hc Hextra.
  destruct Hextra as [ | v v' extra extra' Hv Hextra]; [ exact Hc | ].
  apply Extra_args_around; [ exact Hc | constructor; assumption ].
Qed.

Lemma conts_around_extend (statics' statics'' : static_env)
    (conts conts' : cont_env) :
  statics_extend statics' statics'' ->
  conts_around statics' conts conts' ->
  conts_around statics'' conts conts'.
Proof.
  intros Hextend.
  induction 1 as [ | kc kc' conts conts' [Hkey Hc] _ IH]; constructor;
    [ split; [ exact Hkey | exact (cont_around_extend _ _ Hextend _ _ Hc) ]
    | exact IH ].
Qed.

Lemma resolve_around (statics' : static_env) (k : continuation)
    (conts conts' : cont_env) :
  conts_around statics' conts conts' ->
  cont_around statics' (resolve k conts) (resolve k conts').
Proof.
  unfold resolve.
  induction 1 as [ | [k1 c] [k1' c'] conts conts' [Hk Hc] _ IH]; simpl;
    [ apply Outer_around | ].
  simpl in Hk. subst k1'. destruct (Pos.eqb k k1); [ exact Hc | exact IH ].
Qed.

(* [all] is the whole group, kept apart from the handlers still to be gone
   through. *)
Lemma rec_closures_around_aux (statics' : static_env) (vars vars' : env)
    (conts conts' : cont_env) (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (known : known_constants)
    (q : path) (all : list (continuation * continuation_handler)) :
  env_lifted statics' vars vars' ->
  conts_around statics' conts conts' ->
  known_valid statics' vars' known ->
  rec_handlers_forall (handler_around_ok lets groups known) q all = true ->
  ∀ (handlers : list (continuation * continuation_handler)) (q1 : path),
    rec_handlers_forall (handler_around_ok lets groups known) q1 handlers
    = true ->
    conts_around statics'
      (map (fun '(k, h) => (k, Closure h vars conts all)) handlers)
      (map
         (fun '(k, h) =>
            (k,
             Closure h vars' conts'
               (rec_handlers_map (lift_around_handler lets groups) q all)))
         (rec_handlers_map (lift_around_handler lets groups) q1 handlers)).
Proof.
  intros Hvars Hconts Hvalid Hall.
  induction handlers as [ | [k h] handlers IH]; intros q1;
    cbn [rec_handlers_map rec_handlers_forall map]; [ constructor | ].
  intros Hok. apply andb_true_iff in Hok as [Hh Hrest].
  constructor; [ | exact (IH _ Hrest) ].
  split; [ reflexivity | ]. cbn [snd].
  exact (Closure_around statics' lets groups known (xO q1) h vars vars' conts
           conts' lets groups q all Hvars Hconts Hvalid Hh Hall).
Qed.

Lemma rec_closures_around (statics' : static_env) (vars vars' : env)
    (conts conts' : cont_env) (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (known : known_constants)
    (q : path) (handlers : list (continuation * continuation_handler)) :
  env_lifted statics' vars vars' ->
  conts_around statics' conts conts' ->
  known_valid statics' vars' known ->
  rec_handlers_forall (handler_around_ok lets groups known) q handlers
  = true ->
  conts_around statics' (rec_closures handlers vars conts ++ conts)
    (rec_closures
       (rec_handlers_map (lift_around_handler lets groups) q handlers) vars'
       conts'
     ++ conts').
Proof.
  intros Hvars Hconts Hvalid Hok. apply Forall2_app; [ | exact Hconts ].
  exact (rec_closures_around_aux statics' vars vars' conts conts' lets groups
           known q handlers Hvars Hconts Hvalid Hok handlers q Hok).
Qed.

(* The first state and the second: the second expression is the first
   rewritten, the environments are lifted, and the check passes of the
   expression for constants that are known in the variables of the
   second. *)
Inductive state_around : state -> state -> Prop :=
  | State_around (lets : list (path * lift_to))
      (groups : list (path * list static_binding)) (known : known_constants)
      (at_path : path) (e : expr) (vars vars' : env) (conts conts' : cont_env)
      (statics statics' : static_env) (hp : heap) :
    env_lifted statics' vars vars' ->
    conts_around statics' conts conts' ->
    statics_lifted_out [] statics statics' ->
    known_valid statics' vars' known ->
    around_ok lets groups known at_path e = true ->
    state_around (State e vars conts statics hp)
      (State (lift_around lets groups at_path e) vars' conts' statics' hp).

(* A state and itself rewritten with nothing to do. *)
Lemma state_around_same (e : expr) (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp : heap) :
  env_lifted statics' vars vars' ->
  conts_around statics' conts conts' ->
  statics_lifted_out [] statics statics' ->
  state_around (State e vars conts statics hp)
    (State e vars' conts' statics' hp).
Proof.
  intros Hvars Hconts Hstatics.
  pose proof (State_around [] [] [] root e vars vars' conts conts' statics
    statics' hp Hvars Hconts Hstatics (known_valid_nil statics' vars')
    (around_ok_nothing [] root e))
    as Hstate.
  rewrite lift_around_nothing in Hstate. exact Hstate.
Qed.

(* The values at an exit are lifted, under the static environment that
   the second run has reached. *)
Definition exit_around (last : state) (vs vs' : list value) : Prop :=
  Forall2 (value_lifted (state_statics last)) vs vs'.

(* What the second run answers to a result of the first: nothing is
   asked if the first is undefined; an exit is answered by an exit
   through the same continuation, with values lifted; a step, by a
   step, to a state related. *)
Definition result_around (statics' : static_env)
    (result result' : step_result) : Prop :=
  match result with
  | Undefined => True
  | Exit k exit_vs hp =>
    ∃ exit_vs' : list value,
      result' = Exit k exit_vs' hp
      ∧ Forall2 (value_lifted statics') exit_vs exit_vs'
  | Step t => ∃ t' : state, result' = Step t' ∧ state_around t t'
  end.

(* ======================================================================== *)
(* Calls                                                                    *)
(* ======================================================================== *)

(* A full call: the code is the same in both runs. *)
Lemma call_code_around (closure closure' : value) (c : code)
    (vs vs' : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) :
  value_lifted statics' closure closure' ->
  Forall2 (value_lifted statics') vs vs' ->
  cont_around statics' return_to return_to' ->
  cont_around statics' exn_to exn_to' ->
  statics_lifted_out [] statics statics' ->
  result_around statics'
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp)
    (call_code closure' c vs' args_arity return_arity return_to' exn_to'
       statics' hp).
Proof.
  intros Hclosure Hvs Hreturn Hexn Hstatics.
  unfold result_around, call_code.
  rewrite (kinds_of_values_lifted statics' vs vs' Hvs).
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (code_params_arity c))
        && arity_eqb return_arity (code_result_arity c)));
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [bound | ] eqn:Hbind;
    [ | exact I ].
  destruct (bind_params_lifted statics' (code_params c) vs vs'
              [(code_my_closure c, closure)] [(code_my_closure c, closure')]
              bound Hvs)
    as (bound' & Hbind' & Hbound_vars);
    [ apply env_lifted_cons; [ exact Hclosure | intros x v [=] ]
    | exact Hbind | ].
  rewrite Hbind'. eexists. split; [ reflexivity | ].
  apply state_around_same; [ exact Hbound_vars | | exact Hstatics ].
  constructor; [ split; [ reflexivity | exact Hreturn ] | ].
  constructor; [ split; [ reflexivity | exact Hexn ] | ].
  constructor.
Qed.

(* A call of a given shape, given what the two runs do with a partial
   application. The full call inside an over-application returns to
   what is left of it. *)
Lemma run_call_around
    (return_partial return_partial' : value -> step_result)
    (shape shape' : call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) :
  shape_lifted_out [] statics' shape shape' ->
  cont_around statics' return_to return_to' ->
  cont_around statics' exn_to exn_to' ->
  statics_lifted_out [] statics statics' ->
  (∀ partial partial' : value,
    value_lifted statics' partial partial' ->
    result_around statics' (return_partial partial)
      (return_partial' partial')) ->
  result_around statics'
    (run_call return_partial (Some shape) return_arity return_to exn_to
       statics hp)
    (run_call return_partial' (Some shape') return_arity return_to' exn_to'
       statics' hp).
Proof.
  intros Hshape Hreturn Hexn Hstatics Hpartial.
  destruct Hshape
    as [ closure closure' called c vs vs' args_arity Hclosure Hvs Hbound
       | closure closure' called c vs vs' remaining remaining' args_arity
           remaining_arity Hclosure Hvs Hremaining Hbound
       | partial partial' Hp ];
    simpl.
  - rewrite code_lifted_out_nothing. apply call_code_around; assumption.
  - rewrite code_lifted_out_nothing. apply call_code_around; try assumption.
    apply Over_application_around; assumption.
  - exact (Hpartial partial partial' Hp).
Qed.

(* A call, given what the two runs do when the return continuation is
   applied to a partial application. *)
Lemma call_function_around_from (callee callee' : value) (kind : call_kind)
    (vs vs' : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) :
  value_lifted statics' callee callee' ->
  Forall2 (value_lifted statics') vs vs' ->
  cont_around statics' return_to return_to' ->
  cont_around statics' exn_to exn_to' ->
  statics_lifted_out [] statics statics' ->
  (∀ partial partial' : value,
    value_lifted statics' partial partial' ->
    result_around statics' (apply_cont_value return_to [partial] statics hp)
      (apply_cont_value return_to' [partial'] statics' hp)) ->
  result_around statics'
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp)
    (call_function callee' kind vs' args_arity return_arity return_to'
       exn_to' statics' hp).
Proof.
  intros Hcallee Hvs Hreturn Hexn Hstatics Hpartial.
  unfold call_function.
  destruct (shape_of_call statics callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape;
    [ | exact I ].
  destruct (shape_of_call_lifted_out [] statics statics' callee callee' kind
              vs vs' args_arity return_arity shape Hstatics Hcallee Hvs
              Hshape)
    as (shape' & -> & Hshapes).
  exact (run_call_around _ _ shape shape' return_arity _ _ _ _ statics
           statics' _ Hshapes Hreturn Hexn Hstatics Hpartial).
Qed.

(* Applying a continuation. *)
Lemma apply_cont_value_around (statics' : static_env) (c c' : cont_value) :
  cont_around statics' c c' ->
  ∀ (vs vs' : list value) (statics : static_env) (hp : heap),
    Forall2 (value_lifted statics') vs vs' ->
    statics_lifted_out [] statics statics' ->
    result_around statics' (apply_cont_value c vs statics hp)
      (apply_cont_value c' vs' statics' hp).
Proof.
  induction 1
    as [ k
       | lets groups known at_path [params metadata handler] vars vars' conts
           conts' group_lets group_groups q recursive Hvars Hconts Hvalid Hok
           Hgroup
       | c c' extra extra' Hc IH Hextra
       | remaining remaining' remaining_arity return_arity return_to
           return_to' exn_to exn_to' Hremaining Hreturn IHreturn Hexn _ ];
    intros vs vs' statics hp Hvs Hstatics.
  - simpl. exists vs'. split; [ reflexivity | exact Hvs ].
  - simpl. rewrite (kinds_of_values_lifted statics' vs vs' Hvs).
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs vars) as [bound | ] eqn:Hbind;
      [ | exact I ].
    destruct (bind_params_lifted statics' params vs vs' vars vars' bound Hvs
                Hvars Hbind)
      as (bound' & Hbind' & Hbound_vars).
    rewrite Hbind'. eexists. split; [ reflexivity | ].
    apply (State_around lets groups (forget params known));
      [ exact Hbound_vars
      | exact (rec_closures_around statics' vars vars' conts conts' group_lets
                 group_groups known q recursive Hvars Hconts Hvalid Hgroup)
      | exact Hstatics
      | exact (known_valid_params _ _ _ _ _ _ Hbind' Hvalid) | exact Hok ].
  - simpl. apply IH; [ apply Forall2_app; assumption | exact Hstatics ].
  - rewrite !apply_cont_value_over_application.
    destruct Hvs as [ | v v' vs vs' Hv Hvs]; [ exact I | ].
    destruct Hvs as [ | w w' vs vs' Hw Hvs]; [ | exact I ].
    apply call_function_around_from; try assumption.
    intros partial partial' Hpartial.
    apply IHreturn;
      [ constructor; [ exact Hpartial | constructor ] | exact Hstatics ].
Qed.

(* A call. *)
Lemma call_function_around (callee callee' : value) (kind : call_kind)
    (vs vs' : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) :
  value_lifted statics' callee callee' ->
  Forall2 (value_lifted statics') vs vs' ->
  cont_around statics' return_to return_to' ->
  cont_around statics' exn_to exn_to' ->
  statics_lifted_out [] statics statics' ->
  result_around statics'
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp)
    (call_function callee' kind vs' args_arity return_arity return_to'
       exn_to' statics' hp).
Proof.
  intros Hcallee Hvs Hreturn Hexn Hstatics.
  apply call_function_around_from; try assumption.
  intros partial partial' Hpartial.
  apply (apply_cont_value_around statics' return_to return_to' Hreturn);
    [ constructor; [ exact Hpartial | constructor ] | exact Hstatics ].
Qed.

(* ======================================================================== *)
(* One step                                                                 *)
(* ======================================================================== *)

Lemma result_around_answered (s' : state) (result : step_result) :
  result_around (state_statics s') result (step s') ->
  answered state_around exit_around result s'.
Proof.
  destruct result as [t | k vs | ]; unfold answered; simpl;
    [ | exact (fun H => H) | exact (fun H => H) ].
  intros (t' & Hstep & Ht). exists 0, t'.
  split; [ rewrite Hstep; reflexivity | exact Ht ].
Qed.

(* A step of the first state is answered by the second, which is the
   state of a run that never binds a static name twice and reads no
   variable that has no value: in one step, or in two where a group is
   put before the [Let_cont] that the first executes. *)
Lemma around_step (s s' : state) :
  state_around s s' ->
  static_fault s' = false ->
  reads_unbound_variable s' = false ->
  answered state_around exit_around (step s) s'.
Proof.
  intros
    [lets groups known at_path e vars vars' conts conts' statics statics' hp
            Hvars Hconts Hstatics Hvalid Hok] Hfault Hreads.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ].
  - (* Let *)
    rewrite step_let.
    destruct (eval_named statics vars d) as [v | ] eqn:Hd; [ | exact I ].
    destruct (kind_eqb (kind_of_value v) xkind) eqn:Hkind; [ | exact I ].
    cbn [lift_around around_ok] in Hok |- *.
    apply andb_true_iff in Hok as [Hok_d Hok_body].
    destruct (alias_of lets at_path d) as [[bindings s0] | ] eqn:Halias.
    + (* An alias: the symbol is bound to what the [Let] makes. *)
      unfold alias_of in Halias.
      destruct (lookup at_path lets) as [target | ]; [ | discriminate ].
      pose proof (known_alias target d bindings s0 statics statics' vars vars'
        v known Halias Hok_d Hvalid Hvars Hd)
        as Hv.
      exists 0, (State (lift_around lets groups (xO at_path) body)
                   ((x, Value_symbol s0) :: vars') conts' statics' hp).
      split.
      * simpl. rewrite step_let. cbn [eval_named eval_simple].
        rewrite (kind_of_value_lifted _ _ _ Hv), Hkind. reflexivity.
      * apply (State_around lets groups (forget [x] known));
          [ apply env_lifted_cons; assumption | exact Hconts | exact Hstatics
          | apply known_valid_let; exact Hvalid | exact Hok_body ].
    + destruct (eval_named_lifted statics statics' vars vars' d v
                  (proj1 (proj2 Hstatics)) Hvars Hd)
        as (v' & Hd' & Hv).
      exists 0, (State (lift_around lets groups (xO at_path) body)
                   ((x, v') :: vars') conts' statics' hp).
      split.
      * simpl.
        rewrite step_let, Hd', (kind_of_value_lifted statics' v v' Hv), Hkind.
        reflexivity.
      * apply (State_around lets groups (forget [x] known));
          [ apply env_lifted_cons; assumption | exact Hconts | exact Hstatics
          | apply known_valid_let; exact Hvalid | exact Hok_body ].
  - (* Let_static *)
    cbn [lift_around around_ok] in Hok, Hfault |- *.
    rewrite step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings;
      [ | exact I ].
    unfold static_fault in Hfault. simpl_state.
    apply orb_false_iff in Hfault as [_ Hfault].
    destruct (no_rebinding statics' _ Hfault) as [Hdistinct Hfresh].
    pose proof (eval_static_bindings_lifted_out [] vars vars' bindings statics
      statics' extended Hvars Hstatics)
      as Hgroup.
    rewrite bindings_lift_from_code_nothing_at_all in Hgroup.
    destruct (Hgroup Hdistinct Hfresh Hbindings)
      as (extended' & Hextended & Hlifted).
    pose proof (eval_static_bindings_extend vars' _ statics' extended'
      Hextended) as Hextend.
    exists 0, (State (lift_around lets groups (xO at_path) body) vars' conts'
                 extended' hp).
    split; [ simpl; rewrite step_let_static, Hextended; reflexivity | ].
    apply (State_around lets groups (constants_of_group bindings ++ known));
      [ exact (env_lifted_extend _ _ _ _ Hextend Hvars)
      | exact (conts_around_extend _ _ _ _ Hextend Hconts) | exact Hlifted
      | | exact Hok ].
    apply known_valid_app;
      [ exact (eval_static_bindings_known vars' bindings statics' extended'
                 Hextended)
      | exact (known_valid_extend _ _ _ _ Hextend Hvalid) ].
  - (* Let_cont_non_recursive *)
    cbn [lift_around around_ok] in Hok, Hfault, Hreads |- *.
    cbv zeta in Hok, Hfault, Hreads |- *.
    apply andb_true_iff in Hok as [Hok Hok_body].
    apply andb_true_iff in Hok as [Hok Hok_h].
    apply andb_true_iff in Hok as [Hputable _].
    rewrite step_let_cont_non_recursive.
    destruct (group_at groups at_path) as [ | binding bindings] eqn:Hgroup.
    + (* No group is put. *)
      cbn [constants_of_group flat_map app] in Hok_h, Hok_body.
      eexists 0, _.
      split; [ simpl; rewrite step_let_cont_non_recursive; reflexivity | ].
      apply (State_around lets groups known);
        [ exact Hvars | | exact Hstatics | exact Hvalid | exact Hok_body ].
      constructor; [ | exact Hconts ].
      split; [ reflexivity | ].
      exact (Closure_around statics' lets groups known (xI at_path) h vars
               vars' conts conts' [] [] root [] Hvars Hconts Hvalid Hok_h
               eq_refl).
    + (* A group is put: the second run binds it first. *)
      remember (binding :: bindings) as group eqn:Hgroup_eq.
      unfold static_fault in Hfault. simpl_state. cbn iota in Hfault.
      apply orb_false_iff in Hfault as [_ Hfault].
      destruct (no_rebinding statics' _ Hfault) as [Hdistinct Hfresh].
      destruct (eval_putable_group vars' group statics' Hputable Hdistinct
                  Hfresh)
        as (extended' & Hextended & Hcode).
      { intros y Hy. unfold reads_unbound_variable in Hreads. simpl_state.
        destruct (is_bound y vars') eqn:Hbound; [ reflexivity | ].
        exfalso.
        assert (Hexists :
          existsb (fun x => negb (is_bound x vars'))
            (variables_read
               (Let_static group
                  (Let_cont_non_recursive k
                     (lift_around_handler lets groups (xI at_path) h)
                     (lift_around lets groups (xO at_path) body))))
          = true).
        { apply existsb_exists. exists y.
          split; [ exact Hy | rewrite Hbound; reflexivity ]. }
        exact (eq_true_false_abs _ Hexists Hreads). }
      pose proof (eval_static_bindings_extend vars' _ statics' extended'
        Hextended) as Hextend.
      assert (Hvalid' :
        known_valid extended' vars' (constants_of_group group ++ known)).
      { apply known_valid_app;
          [ exact (eval_static_bindings_known vars' group statics' extended'
                     Hextended)
          | exact (known_valid_extend _ _ _ _ Hextend Hvalid) ]. }
      pose proof (env_lifted_extend _ _ _ _ Hextend Hvars) as Hvars'.
      pose proof (conts_around_extend _ _ _ _ Hextend Hconts) as Hconts'.
      exists 1,
        (State (lift_around lets groups (xO at_path) body) vars'
           ((k,
             Closure (lift_around_handler lets groups (xI at_path) h) vars'
               conts' [])
            :: conts')
           extended' hp).
      split.
      * simpl. rewrite step_let_static, Hextended.
        rewrite step_let_cont_non_recursive. reflexivity.
      * apply (State_around lets groups (constants_of_group group ++ known));
          [ exact Hvars' |
          | exact (statics_lifted_out_extend [] _ _ _ Hstatics Hextend Hcode)
          | exact Hvalid' | exact Hok_body ].
        constructor; [ | exact Hconts' ].
        split; [ reflexivity | ].
        exact (Closure_around extended' lets groups
                 (constants_of_group group ++ known) (xI at_path) h vars vars'
                 conts conts' [] [] root [] Hvars' Hconts' Hvalid' Hok_h
                 eq_refl).
  - (* Let_cont_recursive *)
    cbn [lift_around around_ok] in Hok, Hfault, Hreads |- *.
    cbv zeta in Hok, Hfault, Hreads |- *.
    apply andb_true_iff in Hok as [Hok Hok_body].
    apply andb_true_iff in Hok as [Hok Hok_handlers].
    apply andb_true_iff in Hok as [Hok Hsame].
    apply andb_true_iff in Hok as [Hputable _].
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers) eqn:Hbinds; [ exact I | ].
    apply Bool.eqb_prop in Hsame.
    destruct (group_at groups at_path) as [ | binding bindings] eqn:Hgroup.
    + (* No group is put. *)
      cbn [constants_of_group flat_map app] in Hok_handlers, Hok_body.
      eexists 0, _.
      split;
        [ simpl; rewrite step_let_cont_recursive, Hsame; reflexivity | ].
      apply (State_around lets groups known);
        [ exact Hvars | | exact Hstatics | exact Hvalid | exact Hok_body ].
      exact (rec_closures_around statics' vars vars' conts conts' lets groups
               known (xI at_path) handlers Hvars Hconts Hvalid Hok_handlers).
    + (* A group is put: the second run binds it first. *)
      remember (binding :: bindings) as group eqn:Hgroup_eq.
      unfold static_fault in Hfault. simpl_state. cbn iota in Hfault.
      apply orb_false_iff in Hfault as [_ Hfault].
      destruct (no_rebinding statics' _ Hfault) as [Hdistinct Hfresh].
      destruct (eval_putable_group vars' group statics' Hputable Hdistinct
                  Hfresh)
        as (extended' & Hextended & Hcode).
      { intros y Hy. unfold reads_unbound_variable in Hreads. simpl_state.
        destruct (is_bound y vars') eqn:Hbound; [ reflexivity | ].
        exfalso.
        assert (Hexists :
          existsb (fun x => negb (is_bound x vars'))
            (variables_read
               (Let_static group
                  (Let_cont_recursive
                     (rec_handlers_map (lift_around_handler lets groups)
                        (xI at_path) handlers)
                     (lift_around lets groups (xO at_path) body))))
          = true).
        { apply existsb_exists. exists y.
          split; [ exact Hy | rewrite Hbound; reflexivity ]. }
        exact (eq_true_false_abs _ Hexists Hreads). }
      pose proof (eval_static_bindings_extend vars' _ statics' extended'
        Hextended) as Hextend.
      assert (Hvalid' :
        known_valid extended' vars' (constants_of_group group ++ known)).
      { apply known_valid_app;
          [ exact (eval_static_bindings_known vars' group statics' extended'
                     Hextended)
          | exact (known_valid_extend _ _ _ _ Hextend Hvalid) ]. }
      pose proof (env_lifted_extend _ _ _ _ Hextend Hvars) as Hvars'.
      pose proof (conts_around_extend _ _ _ _ Hextend Hconts) as Hconts'.
      eexists 1, _.
      split.
      * simpl. rewrite step_let_static, Hextended.
        rewrite step_let_cont_recursive, Hsame. reflexivity.
      * apply (State_around lets groups (constants_of_group group ++ known));
          [ exact Hvars' |
          | exact (statics_lifted_out_extend [] _ _ _ Hstatics Hextend Hcode)
          | exact Hvalid' | exact Hok_body ].
        exact (rec_closures_around extended' vars vars' conts conts' lets
                 groups (constants_of_group group ++ known) (xI at_path)
                 handlers Hvars' Hconts' Hvalid' Hok_handlers).
  - (* Apply *)
    cbn [lift_around]. rewrite step_apply.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra;
      [ | exact I ].
    destruct (eval_simple_lifted statics' vars vars' _ callee Hvars Hcallee)
      as (callee' & Hcallee' & Hlifted).
    destruct (eval_simples_lifted statics' vars vars' _ Hvars vs Hargs)
      as (vs' & Hargs' & Hvs).
    destruct (eval_simples_lifted statics' vars vars' _ Hvars extra Hextra)
      as (extra' & Hextra' & Hextras).
    apply result_around_answered. simpl state_statics.
    rewrite step_apply, Hcallee', Hargs', Hextra'.
    apply call_function_around; try assumption.
    + exact (resolve_around statics' _ conts conts' Hconts).
    + exact (with_extra_args_around statics' _ _ extra extra'
               (resolve_around statics' _ conts conts' Hconts) Hextras).
  - (* Apply_cont *)
    cbn [lift_around]. rewrite step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    destruct (eval_simples_lifted statics' vars vars' _ Hvars vs Hargs)
      as (vs' & Hargs' & Hvs).
    apply result_around_answered. simpl state_statics.
    rewrite step_apply_cont, Hargs'. unfold apply_continuation.
    apply (apply_cont_value_around statics' _ _
             (resolve_around statics' _ conts conts' Hconts));
      [ exact Hvs | exact Hstatics ].
  - (* Switch *)
    cbn [lift_around]. rewrite step_switch.
    destruct (eval_simple vars scrutinee) as [v | ] eqn:Hscrutinee;
      [ | exact I ].
    destruct (eval_simple_lifted statics' vars vars' _ v Hvars Hscrutinee)
      as (v' & Hscrutinee' & Hv).
    destruct (naked_immediate_of_value v) as [i | ] eqn:Hi; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    exists 0, (State (Apply_cont ac) vars' conts' statics' hp).
    split.
    + simpl.
      rewrite step_switch, Hscrutinee',
        (naked_immediate_lifted statics' v v' i Hv Hi), Harm.
      reflexivity.
    + exact (State_around lets groups known at_path (Apply_cont ac) vars vars'
               conts conts' statics statics' hp Hvars Hconts Hstatics Hvalid
               eq_refl).
  - (* Invalid *)
    rewrite step_invalid. exact I.
Qed.

(* ======================================================================== *)
(* The run of the unit rewritten                                            *)
(* ======================================================================== *)

(* With no code named and no [Let] lifted out of code, the static
   environments of the two runs are those of FlambdaLiftProofs.v with
   no code named: the same code, and the symbols lifted. *)
Lemma statics_lifted_out_nothing (statics statics' : static_env) :
  statics_lifted_out [] statics statics' -> statics_lifted [] statics statics'.
Proof.
  intros (Hcode & Hsymbols & _). split; [ | exact Hsymbols ].
  intros id. rewrite (Hcode id).
  destruct (lookup id (static_code statics)) as [c | ]; simpl;
    [ | reflexivity ].
  rewrite code_lifted_out_nothing, bind_my_closure_nothing. reflexivity.
Qed.

(* A group that may be put binds no code, and the rewrite leaves the
   code of the unit as it is: so the code of the unit rewritten is
   closed exactly when that of the unit is. *)
Lemma putable_group_code_closed (bindings : list static_binding) :
  putable_group bindings = true ->
  forallb binding_code_closed bindings = true.
Proof.
  unfold putable_group.
  induction bindings as [ | [id c | id | sym const] bindings IH]; simpl;
    try discriminate; [ intros _; reflexivity | ].
  intros H. apply andb_true_iff in H as [_ Hrest]. exact (IH Hrest).
Qed.

Fixpoint lift_around_code_closed (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (known : known_constants)
    (at_path : path) (e : expr) {struct e} :
  around_ok lets groups known at_path e = true ->
  code_closed (lift_around lets groups at_path e) = code_closed e
with lift_around_handler_code_closed (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (known : known_constants)
    (at_path : path) (h : continuation_handler) {struct h} :
  handler_around_ok lets groups known at_path h = true ->
  handler_code_closed (lift_around_handler lets groups at_path h)
  = handler_code_closed h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      try (intros _; reflexivity).
    + simpl. intros H. apply andb_true_iff in H as [_ Hbody].
      destruct (alias_of lets at_path d) as [[bs s] | ]; simpl;
        exact (lift_around_code_closed lets groups _ _ body Hbody).
    + simpl. intros H.
      rewrite (lift_around_code_closed lets groups _ _ body H). reflexivity.
    + simpl. intros H.
      apply andb_true_iff in H as [H Hbody].
      apply andb_true_iff in H as [H Hh].
      apply andb_true_iff in H as [Hput _].
      assert (Hrest :
        code_closed
          (Let_cont_non_recursive k
             (lift_around_handler lets groups (xI at_path) h)
             (lift_around lets groups (xO at_path) body))
        = handler_code_closed h && code_closed body).
      { simpl.
        rewrite (lift_around_handler_code_closed lets groups _ _ h Hh),
          (lift_around_code_closed lets groups _ _ body Hbody).
        reflexivity. }
      revert Hput.
      destruct (group_at groups at_path) as [ | binding bindings];
        intros Hput; [ exact Hrest | ].
      change
        (forallb binding_code_closed (binding :: bindings)
         && code_closed
              (Let_cont_non_recursive k
                 (lift_around_handler lets groups (xI at_path) h)
                 (lift_around lets groups (xO at_path) body))
         = handler_code_closed h && code_closed body).
      rewrite (putable_group_code_closed _ Hput). exact Hrest.
    + simpl. intros H.
      apply andb_true_iff in H as [H Hbody].
      apply andb_true_iff in H as [H Hhandlers].
      apply andb_true_iff in H as [H _].
      apply andb_true_iff in H as [Hput _].
      assert (Hrest :
        code_closed
          (Let_cont_recursive
             (rec_handlers_map (lift_around_handler lets groups) (xI at_path)
                handlers)
             (lift_around lets groups (xO at_path) body))
        = forallb (fun '(_, h) => handler_code_closed h) handlers
          && code_closed body).
      { simpl.
        rewrite (lift_around_code_closed lets groups _ _ body Hbody).
        f_equal. revert Hhandlers. generalize (xI at_path).
        induction handlers as [ | [k1 h1] handlers IH]; intros q Hhandlers;
          [ reflexivity | ].
        cbn [rec_handlers_map rec_handlers_forall forallb] in Hhandlers |- *.
        apply andb_true_iff in Hhandlers as [Hh Hrest].
        rewrite (lift_around_handler_code_closed lets groups _ _ h1 Hh),
          (IH _ Hrest).
        reflexivity. }
      revert Hput.
      destruct (group_at groups at_path) as [ | binding bindings];
        intros Hput; [ exact Hrest | ].
      change
        (forallb binding_code_closed (binding :: bindings)
         && code_closed
              (Let_cont_recursive
                 (rec_handlers_map (lift_around_handler lets groups)
                    (xI at_path) handlers)
                 (lift_around lets groups (xO at_path) body))
         = forallb (fun '(_, h) => handler_code_closed h) handlers
           && code_closed body).
      rewrite (putable_group_code_closed _ Hput). exact Hrest.
  - destruct h as [params metadata handler]. simpl. intros H.
    exact (lift_around_code_closed lets groups _ _ handler H).
Qed.

Section Unit.

Variable u : unit_interface.
Variable lets : list (path * lift_to).
Variable groups : list (path * list static_binding).
Variable e : expr.

(* The static environment and the heap that the two runs start from:
   the environment holds no code and binds no name that the unit
   rewritten binds. *)
Variable statics0 : static_env.
Variable hp0 : heap.

Hypothesis Hwf : well_formed_unit u (lift_around lets groups root e) = true.
Hypothesis Hok : around_ok lets groups [] root e = true.
Hypothesis Hno_code : static_code statics0 = [].
Hypothesis Hfree :
  static_names_free_in statics0 (lift_around lets groups root e) = true.

(* The second state is one that the run of the unit rewritten reaches. *)
Definition reached_around (s' : state) : Prop :=
  ∃ n : nat,
    run n (State (lift_around lets groups root e) [] [] statics0 hp0)
    = Step s'.

(* The states are related and the second is reached. *)
Definition unit_around (s s' : state) : Prop :=
  state_around s s' ∧ reached_around s'.

Lemma unit_around_step (s s' : state) :
  unit_around s s' -> answered unit_around exit_around (step s) s'.
Proof.
  intros (Hrelated & [n Hn]).
  pose proof (well_formed_names_bound_from u _ statics0 hp0 n Hwf Hno_code)
    as Hbound.
  rewrite Hn in Hbound.
  pose proof (around_step s s' Hrelated
    (well_formed_never_faults_from u _ statics0 hp0 n s' Hwf Hno_code Hfree
       Hn)
    Hbound) as Hanswer.
  destruct (step s) as [t | k vs | ]; unfold answered in *;
    [ | exact Hanswer | exact I ].
  destruct Hanswer as (m & t' & Hrun & Ht).
  exists m, t'. split; [ exact Hrun | ].
  split; [ exact Ht | ].
  exists (n + S m). rewrite run_add, Hn. exact Hrun.
Qed.

Lemma unit_around_initial :
  unit_around (State e [] [] statics0 hp0)
    (State (lift_around lets groups root e) [] [] statics0 hp0).
Proof.
  split; [ | exists 0; reflexivity ].
  apply (State_around lets groups []);
    [ intros x v [=] | constructor | | apply known_valid_nil | exact Hok ].
  split; [ intros id; rewrite Hno_code; reflexivity | ].
  split; [ apply symbols_lifted_refl | ].
  intros id c Hlookup. rewrite Hno_code in Hlookup. discriminate Hlookup.
Qed.

(* An exit of the unit is answered by an exit of the unit rewritten
   that nothing that may follow a unit tells apart from it; and the
   unit rewritten ends only if the unit exits or goes wrong. After the
   two exits, what follows is the same term in the two runs, run on
   values lifted under static environments with the same code: the
   simulation of FlambdaLiftProofs.v with no [Let] lifted and no code
   named. No group is put there, so nothing is asked there of the
   variables that are read. *)
Lemma unit_around_worlds :
  refines_world (worlds_observed [])
    (State e [] [] statics0 hp0)
    (State (lift_around lets groups root e) [] [] statics0 hp0).
Proof.
  split.
  - intros w (n & last & Hrun & Hstep & Hstatics).
    destruct
      (stuttering_reaches unit_around exit_around unit_around_step n _ _ last
         unit_around_initial Hrun)
      as (m & last' & Hm & Hrelated & Hreached).
    pose proof (unit_around_step last last' (conj Hrelated Hreached))
      as Hanswer.
    rewrite Hstep in Hanswer. destruct Hanswer as (vs' & Hstep' & Hvs).
    exists (Exit_world (exit_cont w) vs' (state_statics last') (exit_heap w)).
    split; [ exists m, last'; auto | ].
    split; [ reflexivity | ].
    intros c Hobserver. simpl. rewrite <- Hstatics.
    apply
      (observer_applied_lifted [] c (exit_values w) vs' (state_statics last)
         (state_statics last') (exit_heap w) Hobserver Hvs).
    + apply statics_lifted_out_nothing.
      destruct Hrelated; simpl; assumption.
    + intros s' Happly j t' Hj.
      destruct
        (observer_applied_step [] c vs' (state_statics last') (exit_heap w)
           s' Hobserver Happly)
        as (body & vars & Hno & ->).
      exact (never_faults_after u _ statics0 hp0 m last' body vars
               (exit_heap w) j t' Hwf Hno_code Hfree Hm Hno Hj).
  - intros b Hterm Hb.
    destruct (behaves_ended _ b Hterm Hb) as [j Hended].
    destruct
      (stuttering_ends unit_around exit_around unit_around_step j _ _ b
         unit_around_initial Hended)
      as [n [Hwrong | (k & vs & hp & Hexit)]].
    + left. exists n. exact Hwrong.
    + right.
      destruct
        (proj1 (exits_world_behaves (State e [] [] statics0 hp0) k vs hp)
           (ex_intro _ n Hexit))
        as [statics Hw].
      exists (Exit_world k vs statics hp). exact Hw.
Qed.

End Unit.

(* The unit rewritten, run from what may follow a unit, comes to what
   the unit comes to, up to what an exit is compared by: the form of
   FlambdaObserversSpec.v that a pass states, from the one about the
   two exits. *)
Lemma unit_around_observed (u : unit_interface) (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (e : expr)
    (statics0 : static_env) (hp0 : heap) (conts : cont_env) :
  well_formed_unit u (lift_around lets groups root e) = true ->
  around_ok lets groups [] root e = true ->
  static_code statics0 = [] ->
  static_names_free_in statics0 (lift_around lets groups root e) = true ->
  observers [] conts ->
  observed_refines (Step (State e [] conts statics0 hp0))
    (Step (State (lift_around lets groups root e) [] conts statics0 hp0)).
Proof.
  intros Hwf Hok Hno_code Hfree Hobservers.
  pose proof (well_formed_unit_code_closed u _ Hwf) as Hclosed'.
  assert (Hclosed : code_closed e = true).
  { rewrite <- (lift_around_code_closed lets groups [] root e Hok).
    exact Hclosed'. }
  assert (Hstatics : static_code_closed statics0).
  { unfold static_code_closed. rewrite Hno_code. constructor. }
  exact (proj2
           (under_observers_worlds [] statics0 hp0 e _ Hclosed Hclosed'
              Hstatics)
           (unit_around_worlds u lets groups e statics0 hp0 Hwf Hok Hno_code
              Hfree)
           conts Hobservers).
Qed.

(* With the test of the exits. *)
Section Unit_refines.

Variable u : unit_interface.
Variable lets : list (path * lift_to).
Variable groups : list (path * list static_binding).
Variable e : expr.
Variable statics0 : static_env.
Variable hp0 : heap.

Hypothesis Hwf : well_formed_unit u (lift_around lets groups root e) = true.
Hypothesis Hok : around_ok lets groups [] root e = true.
Hypothesis Hno_code : static_code statics0 = [].
Hypothesis Hfree :
  static_names_free_in statics0 (lift_around lets groups root e) = true.

(* The states are related, the second is reached, and the first is one
   of a unit that exits with symbols and constants only. *)
Definition unit_around_plain (s s' : state) : Prop :=
  unit_around lets groups e statics0 hp0 s s' ∧ plain_state s.

(* The values at an exit are lifted, and are constants and symbols: they
   are the same. *)
Lemma unit_around_plain_step (s s' : state) :
  unit_around_plain s s' ->
  answered unit_around_plain (fun _ vs vs' => vs' = vs) (step s) s'.
Proof.
  intros [Hrelated Hplain].
  pose proof
    (unit_around_step u lets groups e statics0 hp0 Hwf Hno_code Hfree s s'
       Hrelated)
    as Hanswer.
  pose proof (plain_step s Hplain) as Hstep.
  destruct (step s) as [t | k vs | ]; unfold answered in *; [ | | exact I ].
  - destruct Hanswer as (m & t' & Hrun & Ht).
    exists m, t'. split; [ exact Hrun | split; assumption ].
  - destruct Hanswer as (vs' & Hexit & Hvs).
    exists vs'. split; [ exact Hexit | ].
    exact (plain_values_lifted _ vs vs' Hstep Hvs).
Qed.

(* The unit rewritten refines the unit, the two run from that static
   environment and that heap. *)
Lemma unit_around_refines :
  exits_plain [] e = true ->
  refines (State e [] [] statics0 hp0)
    (State (lift_around lets groups root e) [] [] statics0 hp0).
Proof.
  intros Hplain.
  apply
    (stuttering_refines unit_around_plain (fun _ vs vs' => vs' = vs)
       unit_around_plain_step).
  - intros last vs vs' Hsame. exact Hsame.
  - split;
      [ exact (unit_around_initial lets groups e statics0 hp0 Hok Hno_code)
      | exact (plain_from e statics0 hp0 Hplain Hno_code) ].
Qed.

End Unit_refines.

Theorem lift_around_unit_program_refines :
  lift_around_unit_program_refines_statement.
Proof.
  intros u lets places e e'. unfold lift_around_unit. cbv zeta.
  set (groups := groups_of lets places e).
  destruct (exits_plain [] e) eqn:Hplain; [ | discriminate ].
  destruct (around_ok lets groups [] root e) eqn:Hok; [ | discriminate ].
  destruct (well_formed_unit u (lift_around lets groups root e)) eqn:Hwf;
    [ | discriminate ].
  intros [= <-]. unfold program_refines.
  exact (unit_around_refines u lets groups e empty_static_env empty_heap Hwf
           Hok eq_refl (free_in_empty _) Hplain).
Qed.

Theorem lift_around_unit_refines_imports :
  lift_around_unit_refines_imports_statement.
Proof.
  intros imports u lets places e e'. unfold lift_around_unit. cbv zeta.
  set (groups := groups_of lets places e).
  destruct (exits_plain [] e) eqn:Hplain; [ | discriminate ].
  destruct (around_ok lets groups [] root e) eqn:Hok; [ | discriminate ].
  destruct (well_formed_unit u (lift_around lets groups root e)) eqn:Hwf;
    [ | discriminate ].
  intros [= <-] Hfree. unfold unit_refines.
  exact (unit_around_refines u lets groups e (imports_statics imports)
           empty_heap Hwf Hok eq_refl Hfree Hplain).
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Lemma group_at_renamed (r : renaming)
    (groups : list (path * list static_binding)) (at_path : path) :
  group_at (rename_groups r groups) at_path
  = map (rename_static_binding r) (group_at groups at_path).
Proof.
  unfold group_at, rename_groups.
  induction groups as [ | [p bindings] groups IH]; [ reflexivity | ].
  cbn [map fst snd]. rewrite !lookup_cons.
  destruct (Pos.eqb at_path p); [ reflexivity | exact IH ].
Qed.

Lemma alias_of_renamed (r : renaming) (lets : list (path * lift_to))
    (at_path : path) (d : named) :
  alias_of lets at_path (rename_named r d)
  = option_map
      (fun '(bindings, s) => (map (rename_static_binding r) bindings, s))
      (alias_of lets at_path d).
Proof.
  unfold alias_of.
  destruct (lookup at_path lets) as [target | ]; [ | reflexivity ].
  apply lift_named_rename.
Qed.

Fixpoint lift_around_renamed (r : renaming) (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (at_path : path) (e : expr)
    {struct e} :
  rename_expr r (lift_around lets groups at_path e)
  = lift_around lets (rename_groups r groups) at_path (rename_expr r e)
with lift_around_handler_renamed (r : renaming)
    (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (at_path : path)
    (h : continuation_handler) {struct h} :
  rename_handler r (lift_around_handler lets groups at_path h)
  = lift_around_handler lets (rename_groups r groups) at_path
      (rename_handler r h).
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [lift_around rename_expr]; try reflexivity.
    + rewrite alias_of_renamed.
      destruct (alias_of lets at_path d) as [[bindings s] | ];
        cbn [option_map rename_expr]; rewrite lift_around_renamed;
        reflexivity.
    + rewrite lift_around_renamed. reflexivity.
    + cbv zeta. rewrite group_at_renamed.
      destruct (group_at groups at_path) as [ | binding bindings];
        cbn [map rename_expr];
        rewrite lift_around_handler_renamed, lift_around_renamed;
        reflexivity.
    + cbv zeta. rewrite group_at_renamed.
      assert (Hhandlers :
        ∀ q : path,
          map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
            (rec_handlers_map (lift_around_handler lets groups) q handlers)
          = rec_handlers_map (lift_around_handler lets (rename_groups r groups))
              q
              (map
                 (fun '(k, h) => (rename_continuation r k, rename_handler r h))
                 handlers)).
      { induction handlers as [ | [k1 h1] handlers IH]; intros q;
          [ reflexivity | ].
        cbn [map rec_handlers_map].
        rewrite lift_around_handler_renamed, IH. reflexivity. }
      destruct (group_at groups at_path) as [ | binding bindings];
        cbn [map rename_expr]; rewrite Hhandlers, lift_around_renamed;
        reflexivity.
  - destruct h as [params metadata handler].
    cbn [lift_around_handler rename_handler].
    rewrite lift_around_renamed. reflexivity.
Qed.

Theorem lift_around_equivariant : lift_around_equivariant_statement.
Proof. exact lift_around_renamed. Qed.

Lemma lift_around_handlers_renamed (r : renaming)
    (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) :
  ∀ (handlers : list (continuation * continuation_handler)) (q : path),
    map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
      (rec_handlers_map (lift_around_handler lets groups) q handlers)
    = rec_handlers_map (lift_around_handler lets (rename_groups r groups)) q
        (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
           handlers).
Proof.
  induction handlers as [ | [k h] handlers IH]; intros q; [ reflexivity | ].
  cbn [map rec_handlers_map].
  rewrite lift_around_handler_renamed, IH. reflexivity.
Qed.

(* A renaming keeps constants apart. *)
Lemma around_simple_injective (r : renaming) (s s' : simple) :
  rename_simple r s = rename_simple r s' -> s = s'.
Proof.
  destruct s as [x | sym | c], s' as [x' | sym' | c']; simpl;
    intros Heq; try discriminate; try exact Heq.
  injection Heq as Heq. f_equal. exact (rename_variable_injective r x x' Heq).
Qed.

Lemma around_simples_injective (r : renaming) :
  ∀ args args' : list simple,
    map (rename_simple r) args = map (rename_simple r) args' -> args = args'.
Proof.
  induction args as [ | s args IH]; intros [ | s' args']; simpl;
    intros Heq; try discriminate; [ reflexivity | ].
  injection Heq as Hs Hargs.
  rewrite (around_simple_injective r s s' Hs), (IH args' Hargs). reflexivity.
Qed.

Lemma around_slots_injective (r : renaming) :
  ∀ slots slots' : list (value_slot * simple),
    map (fun slot => (fst slot, rename_simple r (snd slot))) slots
    = map (fun slot => (fst slot, rename_simple r (snd slot))) slots' ->
    slots = slots'.
Proof.
  induction slots as [ | [slot s] slots IH]; intros [ | [slot' s'] slots'];
    simpl; intros Heq; try discriminate; [ reflexivity | ].
  injection Heq as Hslot Hs Hslots.
  rewrite Hslot, (around_simple_injective r s s' Hs), (IH slots' Hslots).
  reflexivity.
Qed.

Lemma around_const_injective (r : renaming) (const const' : static_const) :
  rename_static_const r const = rename_static_const r const' ->
  const = const'.
Proof.
  destruct const, const'; simpl; intros Heq; try discriminate;
    injection Heq; intros; subst; try reflexivity; f_equal;
    eauto using around_slots_injective, around_simples_injective,
      around_simple_injective.
Qed.

Lemma constant_known_renamed (r : renaming) (known : known_constants)
    (binding : static_binding) :
  constant_known (rename_known_constants r known)
    (rename_static_binding r binding)
  = constant_known known binding.
Proof.
  destruct binding as [id c | id | s const]; try reflexivity.
  cbn [rename_static_binding constant_known]. unfold rename_known_constants.
  induction known as [ | [s1 const1] known IH]; [ reflexivity | ].
  cbn [map existsb fst snd]. rewrite IH. do 2 f_equal.
  destruct (static_const_eq_dec const const1) as [Hsame | Hne].
  - subst const1.
    destruct
      (static_const_eq_dec (rename_static_const r const)
         (rename_static_const r const))
      as [ | Hne'];
      [ reflexivity | contradiction ].
  - destruct
      (static_const_eq_dec (rename_static_const r const)
         (rename_static_const r const1))
      as [Heq | ];
      [ | reflexivity ].
    destruct (Hne (around_const_injective r const const1 Heq)).
Qed.

Lemma all_known_renamed (r : renaming) (known : known_constants)
    (bindings : list static_binding) :
  forallb (constant_known (rename_known_constants r known))
    (map (rename_static_binding r) bindings)
  = forallb (constant_known known) bindings.
Proof.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [map forallb]. rewrite constant_known_renamed, IH. reflexivity.
Qed.

Lemma forget_renamed (r : renaming) (bound : list variable)
    (known : known_constants) :
  forget (map (rename_variable r) bound) (rename_known_constants r known)
  = rename_known_constants r (forget bound known).
Proof.
  unfold forget, rename_known_constants.
  induction known as [ | [s const] known IH]; [ reflexivity | ].
  cbn [map filter fst snd].
  rewrite (free_variables_of_static_const_renamed r const), IH.
  assert (Hexists :
    existsb (fun x => among x (map (rename_variable r) bound))
      (map (rename_variable r) (free_variables_of_static_const const))
    = existsb (fun x => among x bound)
        (free_variables_of_static_const const)).
  { induction (free_variables_of_static_const const) as [ | x xs IHxs];
      [ reflexivity | ].
    cbn [map existsb].
    rewrite (among_map_injective (rename_variable r) (rename_variable_eqb r)),
      IHxs.
    reflexivity. }
  rewrite Hexists.
  destruct
    (existsb (fun x => among x bound) (free_variables_of_static_const const));
    reflexivity.
Qed.

Lemma constants_of_group_renamed (r : renaming)
    (bindings : list static_binding) :
  constants_of_group (map (rename_static_binding r) bindings)
  = rename_known_constants r (constants_of_group bindings).
Proof.
  unfold constants_of_group, rename_known_constants.
  induction bindings as [ | [id c | id | s const] bindings IH];
    cbn [map flat_map rename_static_binding app]; [ reflexivity | | | ];
    rewrite IH; reflexivity.
Qed.

Lemma rename_known_app (r : renaming) (known known' : known_constants) :
  rename_known_constants r known ++ rename_known_constants r known'
  = rename_known_constants r (known ++ known').
Proof. unfold rename_known_constants. rewrite map_app. reflexivity. Qed.

Lemma putable_group_renamed (r : renaming) (bindings : list static_binding) :
  putable_group (map (rename_static_binding r) bindings)
  = putable_group bindings.
Proof.
  unfold putable_group.
  induction bindings as [ | [id c | id | s const] bindings IH];
    cbn [map forallb rename_static_binding]; [ reflexivity | | | ];
    rewrite IH; try reflexivity.
  destruct const; reflexivity.
Qed.

Lemma group_in_scope_renamed (r : renaming) (bindings : list static_binding)
    (e : expr) :
  group_in_scope (map (rename_static_binding r) bindings) (rename_expr r e)
  = group_in_scope bindings e.
Proof.
  unfold group_in_scope.
  assert (Hbindings :
    flat_map free_variables_of_binding (map (rename_static_binding r) bindings)
    = map (rename_variable r) (flat_map free_variables_of_binding bindings)).
  { induction bindings as [ | binding bindings IH]; [ reflexivity | ].
    cbn [map flat_map].
    rewrite map_app, (free_variables_of_binding_renamed r binding), IH.
    reflexivity. }
  rewrite Hbindings, (free_variables_of_expr_renamed r e).
  apply (all_among_map_injective (rename_variable r) (rename_variable_eqb r)).
Qed.

Lemma handlers_bind_statics_renamed (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
       handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH];
    [ reflexivity | ].
  cbn [map existsb rename_handler handler_binds_statics].
  rewrite rename_binds_statics, IH. reflexivity.
Qed.

Fixpoint around_ok_renamed (r : renaming) (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (known : known_constants)
    (at_path : path) (e : expr) {struct e} :
  around_ok lets (rename_groups r groups) (rename_known_constants r known)
    at_path (rename_expr r e)
  = around_ok lets groups known at_path e
with handler_around_ok_renamed (r : renaming)
    (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (known : known_constants)
    (at_path : path) (h : continuation_handler) {struct h} :
  handler_around_ok lets (rename_groups r groups)
    (rename_known_constants r known) at_path (rename_handler r h)
  = handler_around_ok lets groups known at_path h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [around_ok rename_expr]; try reflexivity.
    + rewrite alias_of_renamed.
      change [rename_variable r x] with (map (rename_variable r) [x]).
      rewrite forget_renamed, around_ok_renamed. f_equal.
      destruct (alias_of lets at_path d) as [[bindings s] | ];
        cbn [option_map]; [ apply all_known_renamed | reflexivity ].
    + rewrite constants_of_group_renamed, rename_known_app.
      apply around_ok_renamed.
    + cbv zeta.
      rewrite group_at_renamed, constants_of_group_renamed, rename_known_app,
        putable_group_renamed, handler_around_ok_renamed, around_ok_renamed.
      pose proof
        (group_in_scope_renamed r (group_at groups at_path)
           (Let_cont_non_recursive k h body))
        as Hscope.
      cbn [rename_expr] in Hscope. rewrite Hscope. reflexivity.
    + cbv zeta.
      rewrite group_at_renamed, constants_of_group_renamed, rename_known_app,
        putable_group_renamed, around_ok_renamed.
      pose proof
        (group_in_scope_renamed r (group_at groups at_path)
           (Let_cont_recursive handlers body))
        as Hscope.
      cbn [rename_expr] in Hscope. rewrite Hscope. clear Hscope.
      rewrite <- lift_around_handlers_renamed, !handlers_bind_statics_renamed.
      assert (Hhandlers :
        ∀ (known' : known_constants) (q : path),
          rec_handlers_forall
            (handler_around_ok lets (rename_groups r groups)
               (rename_known_constants r known'))
            q
            (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
               handlers)
          = rec_handlers_forall (handler_around_ok lets groups known') q
              handlers).
      { intros known'.
        induction handlers as [ | [k1 h1] handlers IH]; intros q;
          [ reflexivity | ].
        cbn [map rec_handlers_forall].
        rewrite handler_around_ok_renamed, IH. reflexivity. }
      rewrite Hhandlers. reflexivity.
  - destruct h as [params metadata handler].
    cbn [handler_around_ok rename_handler].
    rewrite forget_renamed. apply around_ok_renamed.
Qed.

Theorem around_ok_equivariant : around_ok_equivariant_statement.
Proof. exact around_ok_renamed. Qed.

(* The groups read off the unit. *)
Fixpoint bindings_of_lets_renamed (r : renaming)
    (lets : list (path * lift_to)) (wanted : list path) (at_path : path)
    (e : expr) {struct e} :
  bindings_of_lets lets wanted at_path (rename_expr r e)
  = map (rename_static_binding r) (bindings_of_lets lets wanted at_path e)
with handler_bindings_of_lets_renamed (r : renaming)
    (lets : list (path * lift_to)) (wanted : list path) (at_path : path)
    (h : continuation_handler) {struct h} :
  handler_bindings_of_lets lets wanted at_path (rename_handler r h)
  = map (rename_static_binding r)
      (handler_bindings_of_lets lets wanted at_path h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr bindings_of_lets]; try reflexivity.
    + rewrite map_app, bindings_of_lets_renamed, alias_of_renamed.
      destruct (among at_path wanted); [ | reflexivity ].
      destruct (alias_of lets at_path d) as [[bindings s] | ]; reflexivity.
    + apply bindings_of_lets_renamed.
    + rewrite map_app, bindings_of_lets_renamed,
        handler_bindings_of_lets_renamed.
      reflexivity.
    + rewrite map_app, bindings_of_lets_renamed. f_equal.
      generalize (xI at_path).
      induction handlers as [ | [k1 h1] handlers IH]; intros q;
        [ reflexivity | ].
      cbn [map rec_handlers_bindings].
      rewrite map_app, handler_bindings_of_lets_renamed, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_bindings_of_lets].
    apply bindings_of_lets_renamed.
Qed.

Theorem groups_of_equivariant : groups_of_equivariant_statement.
Proof.
  intros r lets places e. unfold groups_of, rename_groups.
  rewrite map_map. apply map_ext. intros [p wanted]. cbn [fst snd].
  rewrite bindings_of_lets_renamed. reflexivity.
Qed.
